"""`corn_loss_weighted` must BE `corn_loss` at uniform weights, and must actually weight.

The first property is the load-bearing one: the weighted loss is a generalisation the
render-mode retrain needs, and every earlier head's recipe is stated in terms of `corn_loss`.
If the two disagree at all-ones, "same loss, plus weights" is false and no previous head is
reproducible through the new path.
"""
from __future__ import annotations

import torch

from classifier.model import corn_loss, corn_loss_weighted


def _fixture(k: int = 3, n: int = 24, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, k - 1, generator=g, dtype=torch.float64)
    ranks = torch.randint(0, k, (n,), generator=g)
    return logits, ranks


def test_uniform_weights_reproduce_corn_loss():
    for k in (3, 4):
        logits, ranks = _fixture(k)
        a = corn_loss(logits, ranks, num_classes=k)
        b = corn_loss_weighted(logits, ranks, torch.ones(len(ranks), dtype=torch.float64),
                               num_classes=k)
        assert torch.allclose(a, b, atol=1e-12), (k, float(a), float(b))


def test_constant_weights_are_scale_invariant():
    """A weighted MEAN — doubling every weight must change nothing."""
    logits, ranks = _fixture()
    ones = torch.ones(len(ranks), dtype=torch.float64)
    assert torch.allclose(corn_loss_weighted(logits, ranks, ones),
                          corn_loss_weighted(logits, ranks, 2.0 * ones), atol=1e-12)


def test_duplicated_row_at_half_weight_equals_the_original():
    """THE property the near-dup weighting is for: a row present twice at weight 1/2 must
    give the same loss as the row present once at weight 1."""
    logits, ranks = _fixture(n=16)
    dup_logits = torch.cat([logits, logits])
    dup_ranks = torch.cat([ranks, ranks])
    w = torch.full((2 * len(ranks),), 0.5, dtype=torch.float64)
    assert torch.allclose(corn_loss_weighted(logits, ranks,
                                             torch.ones(len(ranks), dtype=torch.float64)),
                          corn_loss_weighted(dup_logits, dup_ranks, w), atol=1e-12)


def test_zero_weight_row_is_ignored():
    logits, ranks = _fixture(n=20)
    keep = torch.ones(len(ranks), dtype=torch.float64)
    keep[-4:] = 0.0
    both_tasks_survive = all((ranks[:-4] > (k - 1)).sum() > 0 for k in range(2))
    assert both_tasks_survive, "fixture must keep every CORN task populated"
    assert torch.allclose(corn_loss_weighted(logits, ranks, keep),
                          corn_loss(logits[:-4], ranks[:-4]), atol=1e-12)
