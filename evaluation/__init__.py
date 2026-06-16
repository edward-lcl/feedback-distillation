"""Evaluation package.

ProcessBench step-error eval needs only sklearn, so `evaluate_processbench` is
imported eagerly. The text-similarity metrics (BERTScore/ROUGE/BLEU) pull in
heavy optional deps (sentence_transformers, etc.) and are imported lazily — they
load only when actually accessed, so a ProcessBench run never requires them.
"""
from .processbench import evaluate_processbench

_LAZY = {
    "compute_similarity", "extract_gsm8k_answer",
    "evaluate_bertscore", "evaluate_rouge", "evaluate_bleu",
}


def __getattr__(name):
    if name in _LAZY:
        from . import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
