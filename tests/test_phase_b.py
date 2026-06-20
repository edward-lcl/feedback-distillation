"""
Regression tests for the Phase-B tooling: D1 rigor (paired bootstrap) and the
B4 distillation-method score losses + ablation wiring.

Heavy deps (torch / sklearn) are import-skipped so the dep-free CI compile job
still passes; these run locally and on the GPU box where the deps are installed.
"""
import pytest


# --- D1: clustered paired bootstrap on the priv−nogt roc_auc gap -------------

def _synth(rng, n_seq=40, steps=6, priv_better=False):
    yt, ysp, ysn, seq = [], [], [], []
    for s in range(n_seq):
        for _ in range(steps):
            err = int(rng.random() < 0.4)
            yt.append(err); seq.append(s)
            base = rng.normal(0.9 if err else 0.0, 1.0)
            ysn.append(base)
            # priv ranks errors more sharply when priv_better, else ~identical
            ysp.append(rng.normal(1.4 if err else -0.3, 1.0) if priv_better
                       else base + rng.normal(0, 0.01))
    return yt, ysp, ysn, seq


def test_paired_bootstrap_detects_real_gap():
    pytest.importorskip("sklearn")
    import numpy as np
    from experiments.transfer_ci import paired_bootstrap
    yt, ysp, ysn, seq = _synth(np.random.default_rng(0), priv_better=True)
    r = paired_bootstrap(yt, ysp, ysn, seq, n_boot=1000, seed=0)
    assert r["gap_priv_minus_nogt"] > 0
    assert r["ci95"][0] > 0 and r["significant_at_95"] is True
    assert r["p_one_sided_priv_le_nogt"] < 0.05


def test_paired_bootstrap_reports_null():
    pytest.importorskip("sklearn")
    import numpy as np
    from experiments.transfer_ci import paired_bootstrap
    yt, ysp, ysn, seq = _synth(np.random.default_rng(1), priv_better=False)
    r = paired_bootstrap(yt, ysp, ysn, seq, n_boot=1000, seed=0)
    # near-identical scorers → the gap CI must include 0 (not "validated")
    assert r["ci95"][0] <= 0 <= r["ci95"][1]
    assert r["significant_at_95"] is False


# --- B4: score-loss modes + ablation mapping ---------------------------------

def test_score_loss_modes_finite_and_differentiable():
    torch = pytest.importorskip("torch")
    from training.losses import compute_score_loss
    dev = torch.device("cpu")
    for mode in ("mse", "verdict", "soft"):
        for ts in (-0.8, 0.0, 0.7):
            pred = torch.tensor([0.3], requires_grad=True)
            loss = compute_score_loss(pred, ts, mode, dev)
            loss.backward()
            assert torch.isfinite(loss)
            assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_score_loss_rejects_unknown_mode():
    torch = pytest.importorskip("torch")
    from training.losses import compute_score_loss
    with pytest.raises(ValueError):
        compute_score_loss(torch.tensor([0.1]), 0.5, "bogus", torch.device("cpu"))


def test_ablation_table_maps_modes_and_flags():
    pytest.importorskip("torch")
    from experiments.train_slfd import ABLATIONS
    assert ABLATIONS["score_critique"] == ([True, False, True, False], "mse")
    assert ABLATIONS["verdict"] == ([False, False, True, False], "verdict")
    assert ABLATIONS["soft"] == ([False, False, True, False], "soft")
    # logit_kd turns on the logit (KD) flag
    flags, mode = ABLATIONS["logit_kd"]
    assert flags[3] is True and mode == "mse"


def test_set_seed_is_reproducible():
    torch = pytest.importorskip("torch")
    from experiments.train_slfd import set_seed
    set_seed(123); a = torch.rand(5)
    set_seed(123); b = torch.rand(5)
    assert torch.equal(a, b)


def test_logit_kd_requires_a_teacher():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from training.slfd_trainer import SLFDTrainer

    class FakeStudent:
        def __init__(self):
            self.model = nn.Linear(4, 4)
            self.score_head = nn.Linear(4, 1)

    with pytest.raises(ValueError):
        SLFDTrainer(FakeStudent(), teacher=None, dataset=[],
                    loss_flags=[False, False, True, True])
