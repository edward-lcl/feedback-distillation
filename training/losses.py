"""
Loss functions for the knowledge distillation loop.

Three primary losses:
  L_LM    — cross-entropy between student feedback tokens and expert tokens
  L_score — MSE between student score head and teacher score
  L_hidden — cosine similarity between aligned hidden states
  L_logit — KL divergence between token logits (disabled when vocabs differ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Language modelling loss
# ---------------------------------------------------------------------------

def compute_lm_loss(
    teacher_feedback: str,
    student,
    prompt: str,
    answer: str,
    device,
    model_dtype,
    is_math: bool = False,
    examples: list | None = None,
    teacher_score: float = 0.9,
) -> torch.Tensor:
    """Cross-entropy over the feedback token span (student predicted vs teacher label)."""
    input_ids, labels, attention_mask = student.prepare_inputs_and_labels(
        prompt, answer, teacher_feedback, teacher_score, is_math, examples
    )
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    outputs = student.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
    return outputs.loss


# ---------------------------------------------------------------------------
# Hidden-state cosine similarity loss
# ---------------------------------------------------------------------------

def compute_hidden_loss(
    teacher_feedback: str,
    student_tokenizer,
    teacher_tokenizer,
    student,
    teacher,
    align_layer: nn.Module,
    device,
    model_dtype,
) -> torch.Tensor:
    """1 - cosine_similarity between aligned last hidden states."""
    s_inputs = student_tokenizer(teacher_feedback, return_tensors="pt", truncation=True, max_length=256).to(device)
    t_inputs = teacher_tokenizer(teacher_feedback, return_tensors="pt", truncation=True, max_length=256).to(device)

    s_out = student.model(**s_inputs, output_hidden_states=True, return_dict=True)
    with torch.no_grad():
        t_out = teacher.model(**t_inputs, output_hidden_states=True, return_dict=True)

    s_hidden = s_out.hidden_states[-1].mean(dim=1).to(model_dtype)
    t_hidden = t_out.hidden_states[-1].mean(dim=1).to(model_dtype)

    s_aligned = align_layer(s_hidden)
    cos = F.cosine_similarity(s_aligned, t_hidden, dim=-1)
    return (1 - cos).mean()


# ---------------------------------------------------------------------------
# Score regression loss
# ---------------------------------------------------------------------------

def compute_scoring_loss(
    student_score_logit: torch.Tensor,
    teacher_score: float,
    model_dtype,
    device,
) -> torch.Tensor:
    """MSE between student scoring head output and teacher's scalar score."""
    target = torch.tensor([teacher_score], dtype=model_dtype, device=device)
    pred = student_score_logit.to(model_dtype).to(device)
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Logit standardisation loss (disabled when tokenizers differ)
# ---------------------------------------------------------------------------

def compute_logit_standardization(
    student,
    teacher,
    student_tokenizer,
    teacher_tokenizer,
    projection_layer: nn.Module,
    expert_feedback: str,
    max_length: int = 128,
) -> torch.Tensor:
    """Soft KL divergence between student and teacher next-token distributions."""
    device = student.device

    s_inputs = student_tokenizer(expert_feedback, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    t_inputs = teacher_tokenizer(expert_feedback, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    s_out = student.model(**s_inputs, return_dict=True)
    with torch.no_grad():
        t_out = teacher.model(**t_inputs, return_dict=True)

    s_logits = s_out.logits[:, -1, :].float()
    t_logits = t_out.logits[:, -1, :].float()

    s_proj = projection_layer(s_logits)
    min_vocab = min(s_proj.shape[-1], t_logits.shape[-1])
    s_proj = s_proj[..., :min_vocab]
    t_logits = t_logits[..., :min_vocab]

    s_log_probs = F.log_softmax(s_proj, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean")


# ---------------------------------------------------------------------------
# Online logit-KD: soft KL distillation of the teacher's critique distribution
# ---------------------------------------------------------------------------

def compute_logit_kd_loss(
    student,
    teacher,
    problem: str,
    solution_prefix: str,
    step_text: str,
    teacher_feedback: str,
    teacher_score: float,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(teacher ‖ student) over the teacher's critique tokens — the *soft*
    counterpart of the hard-CE LM loss (B4a-online).

    Both models read the SAME sequence (prompt + the teacher's privileged
    critique), built by the student's own tokenizer; we match next-token
    distributions only on the critique span (labels != -100). Requires a
    same-family teacher so the vocabularies align — logits are truncated to the
    shared min-vocab as a safety net. Returns a 0-dim tensor (0.0 if the critique
    span is empty after truncation).
    """
    input_ids, labels, attn = student.prepare_step_inputs_and_labels(
        problem, solution_prefix, step_text, teacher_feedback, teacher_score
    )
    s_dev = student.device
    t_dev = next(teacher.model.parameters()).device

    s_out = student.model(input_ids=input_ids.to(s_dev),
                          attention_mask=attn.to(s_dev), return_dict=True)
    with torch.no_grad():
        t_out = teacher.model(input_ids=input_ids.to(t_dev),
                              attention_mask=attn.to(t_dev), return_dict=True)

    # next-token alignment: position i predicts token i+1
    s_logits = s_out.logits[:, :-1, :].float()
    t_logits = t_out.logits[:, :-1, :].float().to(s_dev)
    mask = (labels[:, 1:] != -100).to(s_dev)            # critique-span positions

    V = min(s_logits.size(-1), t_logits.size(-1))
    s_logits, t_logits = s_logits[..., :V], t_logits[..., :V]

    if mask.sum() == 0:
        return torch.zeros((), device=s_dev, dtype=s_logits.dtype)

    T = temperature
    s_logp = F.log_softmax(s_logits / T, dim=-1)
    t_prob = F.softmax(t_logits / T, dim=-1)
    # per-position KL(teacher ‖ student); kl_div(input=logQ, target=P) = sum P·(logP - logQ)
    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1)   # [B, T-1]
    kl = kl[mask]                                                 # critique tokens only
    # T^2 keeps the gradient magnitude comparable across temperatures (Hinton KD).
    return (T * T) * kl.mean()


# ---------------------------------------------------------------------------
# Combined vectorised loss computation
# ---------------------------------------------------------------------------

def compute_all_losses_vectorized(
    loss_config,
    prompt: str,
    answer: str,
    student,
    teacher,
    student_tokenizer,
    teacher_tokenizer,
    align_layer: nn.Module,
    projection_layer: nn.Module,
    device,
    is_math: bool = False,
    examples: list | None = None,
    model_dtype=torch.float32,
) -> dict:
    """Run one forward pass, compute all active losses, return dict with feedback + loss vector."""
    student_feedback, student_score, student_score_logit = student.generate_feedback(prompt, answer, is_math, examples)
    with torch.no_grad():
        teacher_feedback, teacher_score = teacher.generate_feedback(prompt, answer, is_math, examples)

    loss_names, enabled = loss_config.get_active_losses()
    losses = {}

    if enabled.get("lm_loss"):
        losses["lm_loss"] = compute_lm_loss(teacher_feedback, student, prompt, answer, device, model_dtype, is_math, examples, teacher_score)
    if enabled.get("hidden_loss"):
        losses["hidden_loss"] = compute_hidden_loss(teacher_feedback, student_tokenizer, teacher_tokenizer, student, teacher, align_layer, device, model_dtype)
    if enabled.get("scoring_loss"):
        losses["scoring_loss"] = compute_scoring_loss(student_score_logit, teacher_score, model_dtype, device)
    if enabled.get("logit_loss"):
        losses["logit_loss"] = compute_logit_standardization(student, teacher, student_tokenizer, teacher_tokenizer, projection_layer, teacher_feedback)

    active = [losses[n] for n in loss_names if n in losses]
    loss_vector = torch.stack([l.detach() for l in active]) if active else torch.zeros(1)

    return {
        "student_feedback": student_feedback,
        "teacher_feedback": teacher_feedback,
        "student_score": student_score,
        "teacher_score": teacher_score,
        "student_score_logit": student_score_logit,
        "losses": losses,
        "loss_vector": loss_vector,
    }


def unpack_feedback_and_scores(result: dict) -> tuple[str, str, float, float]:
    return (
        result["student_feedback"],
        result["teacher_feedback"],
        result["student_score"],
        result["teacher_score"],
    )
