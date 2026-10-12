"""Model families: the auto-selection ladder and each model's derived checkpoints.

Every derivation returns None when the family has no such checkpoint for that model.
Derived repos are verified on the Hub at preflight; a missing or gated one only drops the
configs that need it.
"""
from __future__ import annotations

from typing import Optional


def _gb(params_b: float) -> float:
    """BF16 weight size in GiB for a parameter count in billions."""
    return params_b * 2e9 / 2 ** 30


def _qwen3_eagle3(m: str) -> Optional[dict]:
    # Red Hat's speculators-format EAGLE-3 heads (vLLM reads the speculator config directly)
    size = m.rsplit("-", 1)[-1]
    if size in ("8B", "14B", "32B"):
        return {"model": f"RedHatAI/Qwen3-{size}-speculator.eagle3", "method": "eagle3",
                "num_speculative_tokens": 3}
    return None


FAMILIES: dict[str, dict] = {
    "qwen3": dict(
        ladder=[("Qwen/Qwen3-30B-A3B", _gb(30.5)), ("Qwen/Qwen3-14B", _gb(14.8)),
                ("Qwen/Qwen3-8B", _gb(8.2)), ("Qwen/Qwen3-4B", _gb(4.0)),
                ("Qwen/Qwen3-1.7B", _gb(2.0)), ("Qwen/Qwen3-0.6B", _gb(0.75))],
        extra=[("Qwen/Qwen3-32B", _gb(32.8))],
        moe=lambda m: "A3B" in m,
        draft=lambda m: None if m.endswith("0.6B") else
            {"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 4},
        eagle3=_qwen3_eagle3,
        fp8_ckpt=lambda m: f"{m}-FP8",
        w4a16=lambda m: None if ("A3B" in m or m.endswith(("0.6B", "1.7B"))) else f"{m}-AWQ",
        mm_off="none",
    ),
    "gemma4": dict(
        # Google's published BF16 figures carry ~10% over raw weights
        ladder=[("google/gemma-4-26B-A4B-it", 57.7e9 / 1.1 / 2 ** 30),
                ("google/gemma-4-12B-it", 26.7e9 / 1.1 / 2 ** 30),
                ("google/gemma-4-E4B-it", 17.9e9 / 1.1 / 2 ** 30),
                ("google/gemma-4-E2B-it", 11.4e9 / 1.1 / 2 ** 30)],
        extra=[("google/gemma-4-31B-it", 69.9e9 / 1.1 / 2 ** 30)],
        moe=lambda m: "A4B" in m,
        draft=lambda m: {"model": f"{m}-assistant",
                         "num_speculative_tokens": 2 if "E2B" in m else 4},
        eagle3=lambda m: None,
        fp8_ckpt=lambda m: None,
        w4a16=lambda m: None if "26B-A4B" in m else f"{m}-qat-w4a16-ct",
        mm_off="language-model-only",   # skip vision/audio encoders
    ),
}

EPILOG = """
Families (largest first; the first that fits wins):
  qwen3  (default; Apache-2.0, ungated)  Qwen3-30B-A3B (MoE) > 14B > 8B > 4B > 1.7B > 0.6B
         draft: Qwen3-0.6B · EAGLE-3: RedHatAI/Qwen3-{8B,14B,32B}-speculator.eagle3
         FP8 / 4-bit: Qwen's -FP8 / -AWQ repos
  gemma4 (license-gated on HF)           gemma-4-26B-A4B-it (MoE) > 12B > E4B > E2B
         draft: <id>-assistant · 4-bit: <id>-qat-w4a16-ct (none for 26B)
Any other HF id or local path works with --model; family-specific configs are skipped
unless you pass --spec-config / --eagle-config / --w4a16-model.
"""


def family_of(model: str) -> Optional[str]:
    for fam, spec in FAMILIES.items():
        if any(model == m for m, _ in spec["ladder"] + spec["extra"]):
            return fam
    return None


def family(model: str) -> Optional[dict]:
    return FAMILIES.get(family_of(model) or "")


def is_moe(model: str) -> bool:
    fam = family(model)
    if fam:
        return bool(fam["moe"](model))
    low = model.lower()
    return any(t in low for t in ("moe", "mixtral", "-a3b", "-a4b", "deepseek-v", "experts"))


def derived(model: str, kind: str) -> Optional[dict | str]:
    """kind in draft | eagle3 | fp8_ckpt | w4a16."""
    fam = family(model)
    return fam[kind](model) if fam else None
