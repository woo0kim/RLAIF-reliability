"""GPU check that the CUDA-graph sampler matches HF transformers (with and without LoRA).

  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_fastgen_gpu.py -s
"""
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
MODEL = os.environ.get("FASTGEN_TEST_MODEL", "Qwen/Qwen3-0.6B")


def _setup(lora: bool):
    from rlaif.models import chat_prompt, load_model, load_tokenizer

    tok = load_tokenizer(MODEL)
    model = load_model(MODEL, "cuda:0", "float16")
    if lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"),
            autocast_adapter_dtype=True,
        )
        torch.manual_seed(0)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data = (torch.randn_like(p.data.float()) * 0.02)
            elif "lora_A" in n:
                p.data = p.data.float()
        model.eval()
    msgs = [
        [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
        [{"role": "user", "content": "Name three prime numbers greater than 50, and explain how you checked."}],
        [{"role": "user", "content": "Using the numbers [3, 7, 11], make 32 with + - * /."}],
    ]
    prompts = [tok(chat_prompt(tok, m), add_special_tokens=False).input_ids for m in msgs]
    return tok, model, prompts


@pytest.mark.parametrize("lora", [False, True])
def test_greedy_matches_hf(lora):
    from rlaif.fastgen import FastQwen3Sampler
    from rlaif.models import eos_ids, left_pad

    tok, model, prompts = _setup(lora)
    stop = eos_ids(tok)
    t0 = time.time()
    fast = FastQwen3Sampler(model).generate(prompts, 48, stop, tok.pad_token_id, greedy=True, batch_size=4)
    t_fast = time.time() - t0
    ids, mask = left_pad(prompts, tok.pad_token_id, "cuda:0")
    t0 = time.time()
    with torch.no_grad():
        hf = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=48, eos_token_id=stop, pad_token_id=tok.pad_token_id)
    t_hf = time.time() - t0
    hf = hf[:, ids.shape[1] :].tolist()
    agree = []
    for f, h in zip(fast, hf):
        n = 0
        for a, b in zip(f, h):
            if a != b:
                break
            n += 1
        agree.append(n / max(1, min(len(f), 48)))
        print("FAST:", tok.decode(f)[:120].replace("\n", " "))
        print("HF  :", tok.decode(h)[:120].replace("\n", " "))
    print(f"lora={lora} prefix agreement={agree} fast={t_fast:.1f}s hf={t_hf:.1f}s")
    # fp16 near-ties can flip a token late in a sequence; most of each greedy prefix must agree
    assert sum(agree) / len(agree) > 0.6


def test_sampling_distribution_matches_logits():
    """First sampled token frequencies match softmax of HF logits (Gumbel-max correctness)."""
    from rlaif.fastgen import FastQwen3Sampler
    from rlaif.models import eos_ids

    tok, model, prompts = _setup(False)
    p = prompts[1]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([p], device="cuda:0")).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    n = 512
    out = FastQwen3Sampler(model).generate([p] * n, 1, eos_ids(tok), tok.pad_token_id, temperature=1.0, batch_size=128, seed=0)
    first = torch.tensor([o[0] for o in out])
    top = probs.topk(3)
    for pr, i in zip(top.values.tolist(), top.indices.tolist()):
        freq = float((first == i).float().mean())
        print(f"token {i}: p={pr:.3f} freq={freq:.3f}")
        assert abs(freq - pr) < 4 * (pr * (1 - pr) / n) ** 0.5 + 0.02
