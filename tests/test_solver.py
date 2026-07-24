"""Verification 1: closed-form solver correctness (Section 3.5).

Checks that the per-patch closed form (Eq. 9) equals a generic dense linear solve of
the anchored system (Eq. 8), and that the returned features truly minimise the convex
energy (Eq. 7): zero gradient at the solution and higher energy for perturbations.

Run standalone:  python tests/test_solver.py
Or with pytest:  pytest tests/test_solver.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from anoco.solver import assemble_full_laplacian, solve_closed_form, total_energy


def _random_problem(nq=5, nr=8, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_q = torch.randn(nq, d, generator=g, dtype=torch.float64)
    f_r = torch.randn(nr, d, generator=g, dtype=torch.float64)
    w = torch.rand(nq, nr, generator=g, dtype=torch.float64)
    keep = torch.rand(nq, nr, generator=g) < 0.5           # inject sparsity (some zeros)
    w = w * keep
    lam = 0.1 + torch.rand(nq, generator=g, dtype=torch.float64)  # per-query lambda > 0
    return f_q, f_r, w, lam


def test_closed_form_matches_dense_solve():
    f_q, f_r, w, lam = _random_problem()
    f_tilde = solve_closed_form(f_q, f_r, w, lam)

    nq = f_q.shape[0]
    lap = assemble_full_laplacian(w)
    l_qq, l_qr = lap[:nq, :nq], lap[:nq, nq:]
    a_mat = l_qq + torch.diag(lam)                     # (L_qq + Lambda_q), diagonal
    rhs = torch.diag(lam) @ f_q - l_qr @ f_r           # Lambda_q F_q - L_qr F_r
    x = torch.linalg.solve(a_mat, rhs)

    err = (f_tilde - x).abs().max().item()
    assert err < 1e-9, f"closed form disagrees with dense solve: {err}"


def test_solution_is_energy_minimum():
    f_q, f_r, w, lam = _random_problem(seed=1)
    f_tilde = solve_closed_form(f_q, f_r, w, lam)
    e0 = total_energy(f_tilde, f_q, f_r, w, lam)

    g = torch.Generator().manual_seed(2)
    for _ in range(20):
        pert = f_tilde + 0.1 * torch.randn(f_tilde.shape, generator=g, dtype=f_tilde.dtype)
        e1 = total_energy(pert, f_q, f_r, w, lam)
        assert e1 >= e0 - 1e-9, f"found lower energy than closed-form solution: {e1} < {e0}"

    x = f_tilde.clone().requires_grad_(True)
    grad = torch.autograd.grad(total_energy(x, f_q, f_r, w, lam), x)[0]
    gmax = grad.abs().max().item()
    assert gmax < 1e-6, f"gradient not zero at solution: {gmax}"


def test_empty_neighbors_returns_query_unchanged():
    # A query with no neighbours (all-zero weight row) must stay put -> zero drift.
    f_q = torch.randn(3, 4, dtype=torch.float64)
    f_r = torch.randn(5, 4, dtype=torch.float64)
    w = torch.zeros(3, 5, dtype=torch.float64)
    f_tilde = solve_closed_form(f_q, f_r, w, lam=1.0)
    assert torch.allclose(f_tilde, f_q, atol=1e-12)


if __name__ == "__main__":
    test_closed_form_matches_dense_solve()
    test_solution_is_energy_minimum()
    test_empty_neighbors_returns_query_unchanged()
    print("test_solver: ALL PASS")
