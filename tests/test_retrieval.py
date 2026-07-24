"""Verification 2: anchor-driven retrieval rule (Section 3.2, Eq. 1).

Checks anchor selection (argmax similarity), the "longest prefix" stopping behaviour,
and the "mask_all" ablation variant, on a hand-crafted example with a known answer.

Run standalone:  python tests/test_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from anoco.retrieval import anchor_consistent_neighbors, retrieve
from anoco.utils import cosine_sim_matrix


def test_longest_prefix_vs_mask_all():
    # 1 query, 4 references. s already sorted descending: order = [0, 1, 2, 3].
    s = torch.tensor([[0.90, 0.80, 0.70, 0.10]])
    # anchor = ref0 (max s), tau = 0.90.
    tau = torch.tensor([0.90])
    # a_ij = cos(anchor, ref_j). Note ref3 is anchor-compatible (0.99) but sits *after*
    # the violation at ref2 (0.60 <= tau), so the longest-prefix rule must exclude it.
    a = torch.tensor([[1.00, 0.95, 0.60, 0.99]])

    prefix = anchor_consistent_neighbors(s, a, tau, rule="stop_first_violation")
    assert prefix.tolist() == [[True, True, False, False]], prefix.tolist()

    mask_all = anchor_consistent_neighbors(s, a, tau, rule="mask_all")
    assert mask_all.tolist() == [[True, True, False, True]], mask_all.tolist()


def test_anchor_selection_and_pipeline():
    # Two orthogonal reference clusters: A = {ref0, ref1}, B = {ref2, ref3}.
    f_r = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # ref0 (cluster A)
            [0.8, 0.2, 0.0],   # ref1 (cluster A)
            [0.0, 1.0, 0.0],   # ref2 (cluster B)
            [0.0, 0.8, 0.2],   # ref3 (cluster B)
        ]
    )
    f_q = torch.tensor(
        [
            [0.95, 0.02, 0.0],  # query0 -> cluster A
            [0.02, 0.95, 0.0],  # query1 -> cluster B
        ]
    )
    s, mask, anchor_idx, tau = retrieve(f_q, f_r)

    # 1) anchor == argmax cosine similarity (independent recomputation).
    expected = cosine_sim_matrix(f_q, f_r).argmax(dim=1)
    assert torch.equal(anchor_idx, expected), (anchor_idx.tolist(), expected.tolist())

    # 2) each anchor lies in the query's own cluster, and the anchor is a neighbour.
    assert anchor_idx[0].item() in (0, 1)
    assert anchor_idx[1].item() in (2, 3)
    assert mask[0, anchor_idx[0]] and mask[1, anchor_idx[1]]

    # 3) no cross-cluster edges (bipartite graph must not link to the far cluster).
    assert not mask[0, 2].item() and not mask[0, 3].item()
    assert not mask[1, 0].item() and not mask[1, 1].item()


def test_exact_match_query_has_no_neighbors():
    # A query identical to a reference has tau = 1, so a_ij > 1 is never true -> empty set.
    f_r = torch.eye(3)
    f_q = torch.tensor([[1.0, 0.0, 0.0]])
    _, mask, anchor_idx, tau = retrieve(f_q, f_r)
    assert anchor_idx.item() == 0
    assert abs(tau.item() - 1.0) < 1e-6
    assert mask.sum().item() == 0


if __name__ == "__main__":
    test_longest_prefix_vs_mask_all()
    test_anchor_selection_and_pipeline()
    test_exact_match_query_has_no_neighbors()
    print("test_retrieval: ALL PASS")
