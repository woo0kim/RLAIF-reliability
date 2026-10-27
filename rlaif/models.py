"""Model loading, batched sampling and per-token log-probabilities.

Plain PyTorch + HF transformers: the target GPUs (Maxwell, sm_52) support
neither vLLM nor Triton, and fp16 runs as fast as fp32 there while bf16 is
emulated. Frozen weights are therefore stored in fp16; LoRA adapters stay fp32.
"""
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, device: str, dtype: str = "float16"):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPES[dtype], attn_implementation="sdpa")
    # Drop the checkpoint's sampling defaults (top_k=20, top_p=0.95): HF merges them into any
    # generation config we pass, which would make samples disagree with the policy's log-probs.
    model.generation_config = GenerationConfig(
        bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=model.generation_config.pad_token_id,
    )
    model.to(device)
    model.eval()
    return model


def is_oom(e: BaseException) -> bool:
    """CUDA OOM; with the server's driver/NVML mismatch it surfaces as an NVML INTERNAL ASSERT."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or "NVML" in str(e) or "out of memory" in str(e)


def eos_ids(tok) -> List[int]:
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def chat_prompt(tok, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def left_pad(seqs: Sequence[Sequence[int]], pad_id: int, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, L - len(s) :] = torch.tensor(s, dtype=torch.long)
            mask[i, L - len(s) :] = 1
    return ids.to(device), mask.to(device)


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def generate(
    model,
    tok,
    prompts: List[str],
    n: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    do_sample: bool = True,
    batch_size: int = 64,
    seed: Optional[int] = None,
    fast: bool = True,
) -> List[List[Dict]]:
    """Sample n completions per prompt. Returns out[i][j] = {text, ids, finished, n_tokens}.

    Completion ids include the terminating EOS token (so training also teaches
    the policy to stop); `finished` is False when max_new_tokens was hit.
    `fast` uses the CUDA-graph sampler in fastgen.py; otherwise HF generate.
    """
    device = next(model.parameters()).device
    stop = eos_ids(tok)
    if fast:
        from .fastgen import FastQwen3Sampler

        enc = [tok(p, add_special_tokens=False).input_ids for p in prompts]
        flat_ids = [e for e in enc for _ in range(n)]
        gen = FastQwen3Sampler(model).generate(
            flat_ids,
            max_new_tokens=max_new_tokens,
            stop_ids=stop,
            pad_id=tok.pad_token_id,
            temperature=temperature,
            greedy=not do_sample,
            batch_size=batch_size,
            seed=seed,
        )
        out = []
        for i in range(len(prompts)):
            group = []
            for c in gen[i * n : (i + 1) * n]:
                finished = len(c) > 0 and c[-1] in stop
                body = c[:-1] if finished else c
                group.append({"text": tok.decode(body, skip_special_tokens=True), "ids": c, "finished": finished, "n_tokens": len(c)})
            out.append(group)
        return out
    gc = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=1.0 if do_sample else None,
        top_k=0 if do_sample else None,
        eos_token_id=stop,
        pad_token_id=tok.pad_token_id,
    )
    enc = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    jobs = sorted([(i, j) for i in range(len(prompts)) for j in range(n)], key=lambda x: -len(enc[x[0]]))
    out: List[List[Optional[Dict]]] = [[None] * n for _ in prompts]

    def run(chunk, bs_seed):
        ids, mask = left_pad([enc[i] for i, _ in chunk], tok.pad_token_id, device)
        if bs_seed is not None:
            torch.manual_seed(bs_seed)
        try:
            gen = model.generate(input_ids=ids, attention_mask=mask, generation_config=gc)
        except RuntimeError as e:
            if not is_oom(e) or len(chunk) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(chunk) // 2
            run(chunk[:h], None if bs_seed is None else bs_seed + 1)
            run(chunk[h:], None if bs_seed is None else bs_seed + 2)
            return
        comp = gen[:, ids.shape[1] :].tolist()
        for (i, j), c in zip(chunk, comp):
            cut = next((k for k, t in enumerate(c) if t in stop), None)
            finished = cut is not None
            c = c[: cut + 1] if finished else c
            body = c[:-1] if finished else c
            out[i][j] = {
                "text": tok.decode(body, skip_special_tokens=True),
                "ids": c,
                "finished": finished,
                "n_tokens": len(c),
            }

    for b in range(0, len(jobs), batch_size):
        run(jobs[b : b + batch_size], None if seed is None else seed + b)
        torch.cuda.empty_cache()
    return out


def build_lm_batch(prompt_ids: List[List[int]], comp_ids: List[List[int]], pad_id: int, device):
    """[left-padded prompt | right-padded completion] so every completion starts at column P."""
    P = max(len(p) for p in prompt_ids)
    C = max(len(c) for c in comp_ids)
    B = len(prompt_ids)
    ids = torch.full((B, P + C), pad_id, dtype=torch.long)
    mask = torch.zeros((B, P + C), dtype=torch.long)
    cmask = torch.zeros((B, C), dtype=torch.float32)
    for b, (p, c) in enumerate(zip(prompt_ids, comp_ids)):
        ids[b, P - len(p) : P] = torch.tensor(p)
        mask[b, P - len(p) : P] = 1
        ids[b, P : P + len(c)] = torch.tensor(c)
        mask[b, P : P + len(c)] = 1
        cmask[b, : len(c)] = 1.0
    return ids.to(device), mask.to(device), cmask.to(device), P, C


def completion_logprobs(model, ids, mask, P: int, C: int, temperature: float = 1.0) -> torch.Tensor:
    """log pi(y_t | x, y_<t) for every completion position: [B, C] (fp32)."""
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=position_ids_from_mask(mask),
        use_cache=False,
        logits_to_keep=C + 1,
    )
    logits = out.logits[:, :-1, :]
    targets = ids[:, P:]
    rows = []
    for b in range(logits.shape[0]):  # row-wise to bound fp32 memory over the 152k vocab
        lg = logits[b].float()
        if temperature != 1.0:
            lg = lg / temperature
        rows.append(lg.gather(-1, targets[b, :, None]).squeeze(-1) - torch.logsumexp(lg, dim=-1))
    return torch.stack(rows)
