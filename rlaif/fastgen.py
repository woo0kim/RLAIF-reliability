"""Fast sampling for Qwen3 on old GPUs: static KV cache + one CUDA graph per decode step.

HF `generate` is launch-bound on our Maxwell GPUs (~0.12 s of CPU work per
decode step, on an oversubscribed host), and Triton/torch.compile are not
available for sm_52. This module re-implements the Qwen3 decoder forward on top
of the *same parameter tensors* as the HF model (including unmerged PEFT LoRA
adapters, so optimizer updates are seen without copying) and captures the
single-token decode step in a CUDA graph.

Sampling uses the Gumbel-max trick (exact sampling from softmax(logits / T)),
which is graph-capturable. Verified against HF in tests/test_fastgen_gpu.py.
"""
import gc
import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .models import is_oom


def _lin(module):
    """(W, bias, A, B, scale) for nn.Linear or a PEFT LoRA Linear (active adapter, unmerged)."""
    if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
        base = module.base_layer
        names = [n for n in module.active_adapters if n in module.lora_A]
        if names and not getattr(module, "disable_adapters", False) and not getattr(module, "merged", False):
            n = names[0]
            return (base.weight, base.bias, module.lora_A[n].weight, module.lora_B[n].weight, module.scaling[n])
        return (base.weight, base.bias, None, None, 0.0)
    return (module.weight, module.bias, None, None, 0.0)


def _apply(x, lin):
    W, b, A, B, s = lin
    y = F.linear(x, W, b)
    if A is not None:
        y = y + (F.linear(F.linear(x.to(A.dtype), A), B) * s).to(y.dtype)
    return y


def _rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class FastQwen3Sampler:
    def __init__(self, model):
        core = model
        if hasattr(model, "peft_config"):  # PeftModel -> LoraModel -> Qwen3ForCausalLM
            core = model.base_model.model
        cfg = core.config
        assert cfg.model_type == "qwen3", f"FastQwen3Sampler supports qwen3, got {cfg.model_type}"
        self.core = core
        self.H = cfg.num_attention_heads
        self.KV = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.rep = self.H // self.KV
        self.eps = cfg.rms_norm_eps
        self.nl = cfg.num_hidden_layers
        rot = core.model.rotary_emb
        self.inv_freq = rot.inv_freq.float()
        self.rope_scale = float(getattr(rot, "attention_scaling", 1.0))
        self.device = core.model.embed_tokens.weight.device
        self.dtype = core.model.embed_tokens.weight.dtype

    # ---- weights (re-read every call so adapter state changes are honoured) ----
    def _weights(self):
        m = self.core.model
        layers = []
        for L in m.layers:
            a, f = L.self_attn, L.mlp
            layers.append(
                dict(
                    ln1=L.input_layernorm.weight,
                    ln2=L.post_attention_layernorm.weight,
                    q=_lin(a.q_proj),
                    k=_lin(a.k_proj),
                    v=_lin(a.v_proj),
                    o=_lin(a.o_proj),
                    qn=a.q_norm.weight,
                    kn=a.k_norm.weight,
                    gate=_lin(f.gate_proj),
                    up=_lin(f.up_proj),
                    down=_lin(f.down_proj),
                )
            )
        return m.embed_tokens.weight, layers, m.norm.weight, self.core.lm_head.weight

    def _rope(self, pos):  # pos [B, T] long -> cos, sin [B, 1, T, D]
        freqs = pos.float()[..., None] * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.rope_scale).to(self.dtype)[:, None], (emb.sin() * self.rope_scale).to(self.dtype)[:, None]

    def _qkv(self, h, w, pos):
        B, T, _ = h.shape
        q = _rms(_apply(h, w["q"]).view(B, T, self.H, self.D), w["qn"], self.eps).transpose(1, 2)
        k = _rms(_apply(h, w["k"]).view(B, T, self.KV, self.D), w["kn"], self.eps).transpose(1, 2)
        v = _apply(h, w["v"]).view(B, T, self.KV, self.D).transpose(1, 2)
        cos, sin = self._rope(pos)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin, v

    def _mlp(self, x, w):
        h = _rms(x, w["ln2"], self.eps)
        return x + _apply(F.silu(_apply(h, w["gate"])) * _apply(h, w["up"]), w["down"])

    # ---- prefill: whole left-padded prompt, fills cache columns [0, P) ------------
    def _prefill(self, W, ids, start, kc, vc):
        embed, layers, norm, head = W
        B, P = ids.shape
        cols = torch.arange(P, device=self.device)
        pos = (cols[None, :] - start[:, None]).clamp(min=0)
        valid = cols[None, :] >= start[:, None]  # [B, P] key validity
        causal = cols[None, :, None] >= cols[None, None, :]  # [1, Tq, Tk]
        mask = (causal & valid[:, None, :]) | torch.eye(P, dtype=torch.bool, device=self.device)[None]
        mask = mask[:, None]  # [B, 1, P, P]; pad queries attend to themselves only (avoids NaN rows)
        x = embed[ids]
        for li, w in enumerate(layers):
            q, k, v = self._qkv(_rms(x, w["ln1"], self.eps), w, pos)
            kc[li][:, :, :P] = k
            vc[li][:, :, :P] = v
            kk = k.repeat_interleave(self.rep, dim=1)
            vv = v.repeat_interleave(self.rep, dim=1)
            att = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            x = x + _apply(att.transpose(1, 2).reshape(B, P, self.H * self.D), w["o"])
            x = self._mlp(x, w)
        return F.linear(_rms(x[:, -1:], norm, self.eps), head)[:, 0].float()

    # ---- one decode step at column `cur` (graph-captured) -------------------------
    def _decode(self, W, tok, pos, cur, keymask, kc, vc, temperature, greedy):
        embed, layers, norm, head = W
        B = tok.shape[0]
        x = embed[tok][:, None]  # [B, 1, h]
        allowed = keymask & (self.cols[None, :] <= cur)  # [B, Lmax]
        neg = torch.zeros(allowed.shape, dtype=torch.float32, device=self.device).masked_fill(~allowed, float("-inf"))
        scale = 1.0 / math.sqrt(self.D)
        for li, w in enumerate(layers):
            q, k, v = self._qkv(_rms(x, w["ln1"], self.eps), w, pos[:, None])
            kc[li].index_copy_(2, cur.view(1), k)
            vc[li].index_copy_(2, cur.view(1), v)
            qg = q.view(B, self.KV, self.rep, self.D)
            s = torch.matmul(qg, kc[li].transpose(-1, -2)).float() * scale + neg[:, None, None, :]
            p = torch.softmax(s, dim=-1).to(self.dtype)
            att = torch.matmul(p, vc[li]).reshape(B, 1, self.H * self.D)
            x = x + _apply(att, w["o"])
            x = self._mlp(x, w)
        logits = F.linear(_rms(x, norm, self.eps), head)[:, 0].float()
        return self._pick(logits, temperature, greedy)

    @staticmethod
    def _pick(logits, temperature, greedy):
        if greedy:
            return logits.argmax(-1)
        u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-7)
        return (logits / temperature - torch.log(-torch.log(u))).argmax(-1)

    # ---- public API ----------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        stop_ids: Sequence[int],
        pad_id: int,
        temperature: float = 1.0,
        greedy: bool = False,
        batch_size: int = 32,
        seed: Optional[int] = None,
        check_every: int = 16,
    ) -> List[List[int]]:
        """Returns the generated ids for each prompt (truncated after the first stop id, inclusive)."""
        if seed is not None:
            torch.manual_seed(seed)
        n = len(prompt_ids)
        order = sorted(range(n), key=lambda i: -len(prompt_ids[i]))
        results: List[Optional[List[int]]] = [None] * n
        Pmax = max(len(p) for p in prompt_ids)
        while True:
            try:
                with torch.cuda.device(self.device):  # streams and graph capture follow the current device
                    self._run(prompt_ids, order, results, Pmax, max_new_tokens, stop_ids, pad_id, temperature, greedy, batch_size, check_every)
                return results
            except RuntimeError as e:
                if not is_oom(e) or batch_size <= 1:
                    raise
            # outside the except block, so the failed attempt's tensors are no longer referenced
            self._free()
            batch_size //= 2
            print(f"[fastgen] OOM, retrying with batch_size={batch_size}", flush=True)

    def _free(self):
        for a in ("graph", "kc", "vc", "s_tok", "s_pos", "s_cur", "s_keymask", "s_out", "cols"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect()
        torch.cuda.empty_cache()

    def _run(self, prompt_ids, order, results, Pmax, max_new, stop_ids, pad_id, temperature, greedy, B, check_every):
        dev = self.device
        W = self._weights()
        Lmax = Pmax + max_new
        self.cols = torch.arange(Lmax, device=dev)
        self.kc = [torch.zeros(B, self.KV, Lmax, self.D, dtype=self.dtype, device=dev) for _ in range(self.nl)]
        self.vc = [torch.zeros(B, self.KV, Lmax, self.D, dtype=self.dtype, device=dev) for _ in range(self.nl)]
        self.s_tok = torch.zeros(B, dtype=torch.long, device=dev)
        self.s_pos = torch.zeros(B, dtype=torch.long, device=dev)
        self.s_cur = torch.tensor(Lmax - 1, dtype=torch.long, device=dev)
        self.s_keymask = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
        stop = torch.tensor(list(stop_ids), device=dev)

        # capture the decode step. Warm-up runs really execute: give them a valid key mask, else
        # they write NaN K/V into the cache, and 0 * NaN poisons later (masked) attention.
        self.s_keymask.fill_(True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._decode(W, self.s_tok, self.s_pos, self.s_cur, self.s_keymask, self.kc, self.vc, temperature, greedy)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.s_out = self._decode(W, self.s_tok, self.s_pos, self.s_cur, self.s_keymask, self.kc, self.vc, temperature, greedy)
        for t in self.kc + self.vc:
            t.zero_()

        try:
            for c in range(0, len(order), B):
                idx = order[c : c + B]
                real = len(idx)
                idx = idx + [idx[-1]] * (B - real)  # pad the chunk to the captured batch size
                lens = [len(prompt_ids[i]) for i in idx]
                P = max(lens)
                ids = torch.full((B, P), pad_id, dtype=torch.long)
                for r, i in enumerate(idx):
                    ids[r, P - lens[r] :] = torch.tensor(prompt_ids[i], dtype=torch.long)
                ids = ids.to(dev)
                start = torch.tensor([P - l for l in lens], device=dev)
                logits = self._prefill(W, ids, start, self.kc, self.vc)
                tok = self._pick(logits, temperature, greedy)
                self.s_keymask.copy_(self.cols[None, :] >= start[:, None])
                out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
                done = torch.zeros(B, dtype=torch.bool, device=dev)
                steps = 0
                for s in range(max_new):
                    out[:, s] = torch.where(done, torch.full_like(tok, pad_id), tok)
                    done |= torch.isin(tok, stop)
                    steps = s + 1
                    if s == max_new - 1:
                        break
                    if (s + 1) % check_every == 0 and bool(done[:real].all()):
                        break
                    self.s_tok.copy_(tok)
                    self.s_pos.copy_(P + s - start)
                    self.s_cur.fill_(P + s)
                    self.graph.replay()
                    tok = self.s_out.clone()
                out = out[:real, :steps].tolist()
                stop_set = set(stop_ids)
                for r in range(real):
                    seq = out[r]
                    cut = next((k for k, t in enumerate(seq) if t in stop_set), None)
                    results[idx[r]] = seq[: cut + 1] if cut is not None else seq
        finally:
            self._free()
