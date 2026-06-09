"""Evaluation metrics for feedback distillation experiments."""

import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rs
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from detoxify import Detoxify

_embedder: SentenceTransformer | None = None
_detoxify: Detoxify | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_detoxify() -> Detoxify:
    global _detoxify
    if _detoxify is None:
        _detoxify = Detoxify("original")
    return _detoxify


# ---------------------------------------------------------------------------
# Feedback similarity
# ---------------------------------------------------------------------------

def compute_similarity(fb1: str, fb2: str) -> float:
    emb = _get_embedder()
    v1 = torch.tensor(emb.encode([fb1]), dtype=torch.float32)
    v2 = torch.tensor(emb.encode([fb2]), dtype=torch.float32)
    return float(F.cosine_similarity(v1, v2).item())


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> int | float | None:
    """Extract #### <number> from GSM8K ground truth or model output."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        s = m.group(1)
        return float(s) if "." in s else int(s)
    return None


def extract_numeric(text: str) -> float | None:
    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", "").strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_correct_gsm8k(prediction: str, ground_truth: str) -> bool:
    pred = extract_numeric(prediction)
    gt = extract_gsm8k_answer(ground_truth)
    if pred is None or gt is None:
        return False
    return abs(pred - gt) < 1e-6


# ---------------------------------------------------------------------------
# Text-quality metrics
# ---------------------------------------------------------------------------

def evaluate_bertscore(predictions: list[str], references: list[str]) -> dict[str, float]:
    P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}


def evaluate_rouge(prediction: str, reference: str) -> dict[str, float]:
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {"rougeL_f": scores["rougeL"].fmeasure}


def evaluate_bleu(prediction: str, reference: str) -> float:
    ref_tokens = [reference.split()]
    hyp_tokens = prediction.split()
    sf = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=sf)


def evaluate_toxicity(text: str) -> float:
    det = _get_detoxify()
    return float(det.predict(text)["toxicity"])
