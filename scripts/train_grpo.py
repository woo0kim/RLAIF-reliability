"""LoRA GRPO of the policy against a frozen judge. Resumes from the latest checkpoint in out_dir.

  CUDA_VISIBLE_DEVICES=0,1 python scripts/train_grpo.py --config configs/countdown.yaml \
      judge=Qwen/Qwen3-0.6B judge_mode=standard
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlaif.grpo import GRPOTrainer  # noqa: E402
from rlaif.utils import load_config, short_model_name  # noqa: E402


def default_run_name(cfg) -> str:
    name = f"{short_model_name(cfg['judge'])}_{cfg['judge_mode']}_{cfg['pool']}_s{cfg['seed']}"
    return name.replace("qwen3-", "j")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    cfg.setdefault("out_dir", os.path.join("runs", cfg["task"], default_run_name(cfg)))
    import torch

    torch.set_num_threads(4)  # the host CPU is oversubscribed; extra intra-op threads only contend
    GRPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
