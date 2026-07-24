"""ANoCo orchestrator: end-to-end training-free anomaly scoring.

Pipeline (Sections 3.2-3.6):
    retrieve anchor-consistent neighbours  ->  build bipartite edge weights
    ->  closed-form anchored Laplacian solve  ->  non-conformity energy / map / score.

``score_features`` is fully backbone-agnostic (operates on patch features), while
``run_image`` lazily instantiates the DINOv3-L/16 extractor for image inputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from .config import ANoCoConfig
from .graph import edge_weights, norm_compatibility
from .retrieval import retrieve
from .scoring import energy_to_map, image_score, patch_energy
from .solver import solve_closed_form


class ANoCo:
    def __init__(self, config: Optional[ANoCoConfig] = None):
        self.cfg = config or ANoCoConfig()
        self._extractor = None

    @property
    def device(self) -> str:
        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.cfg.device

    @torch.no_grad()
    def score_features(
        self,
        f_q: torch.Tensor,
        f_r: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]] = None,
        out_hw: Optional[Tuple[int, int]] = None,
        chunk: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Score query patch features against a fixed reference pool.

        Args:
            f_q: (N_q, d) query patch features.
            f_r: (N_r, d) reference patch features (all K reference views concatenated).
            grid_hw: (H, W) query patch grid; if given, an anomaly map is produced.
            out_hw: optional output resolution for the anomaly map (bilinear upsample).
            chunk: if >0, process query patches in chunks of this size to cap GPU memory.
                Mathematically identical (each patch is scored independently); the large
                (N_q, N_r) intermediates (sim/neighbor_mask/w) are then returned as None.
        """
        cfg = self.cfg
        dev = self.device
        f_q = f_q.to(dev)
        f_r = f_r.to(dev)

        if chunk and chunk > 0 and f_q.shape[0] > chunk:
            return self._score_chunked(f_q, f_r, grid_hw, out_hw, int(chunk))

        s, mask, anchor_idx, tau = retrieve(f_q, f_r, rule=cfg.prefix_rule,
                                             topk=getattr(cfg, 'retrieve_topk', 200))
        alpha = norm_compatibility(f_q, f_r, eps=cfg.eps)
        w = edge_weights(s, mask, alpha, clamp_negative=cfg.clamp_negative_weights)
        f_tilde = solve_closed_form(f_q, f_r, w, lam=cfg.lam, eps=cfg.eps)
        energy = patch_energy(f_q, f_tilde, metric=cfg.score_metric, eps=cfg.eps)
        score = image_score(energy, reduction=cfg.image_reduction)

        amap = None
        if grid_hw is not None:
            amap = energy_to_map(
                energy, grid_hw, out_hw, cfg.gaussian_kernel, cfg.gaussian_sigma
            )

        return {
            "score": score,
            "patch_energy": energy,
            "anomaly_map": amap,
            "f_tilde": f_tilde,
            "w": w,
            "neighbor_mask": mask,
            "sim": s,
            "anchor_idx": anchor_idx,
            "tau": tau,
            "num_neighbors": mask.sum(dim=1),
        }

    def _score_chunked(
        self,
        f_q: torch.Tensor,
        f_r: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]],
        out_hw: Optional[Tuple[int, int]],
        chunk: int,
    ) -> Dict[str, torch.Tensor]:
        """Memory-capped scoring: loop over query-patch chunks.

        When retrieve_topk > 0, uses retrieve_tiled (tiled similarity + grouped bmm)
        to avoid materializing full (chunk, N_r) matrices. Precision identical."""
        cfg = self.cfg
        topk = getattr(cfg, 'retrieve_topk', 0)
        n_q = f_q.shape[0]
        energies, f_tildes, anchors, taus, nneigh = [], [], [], [], []

        for start in range(0, n_q, chunk):
            fq_c = f_q[start:start + chunk]

            if topk > 0:
                # === Optimized path: tiled + sparse ===
                from .retrieval import retrieve_tiled, sparse_solve
                s, mask, anchor_idx, tau = retrieve_tiled(
                    fq_c, f_r, rule=cfg.prefix_rule, eps=cfg.eps, topk=topk)

                # Extract top-K info for sparse solve
                K = min(topk, f_r.shape[0])
                s_topk_vals, s_topk_idx = s.topk(K, dim=1, largest=True, sorted=True)

                # Compute alpha ONLY for top-K refs (avoids full N_q×N_r matrix)
                fq_norm = fq_c.norm(dim=1, keepdim=True)  # (N_q, 1)
                # Gather f_r norms for top-K refs using grouped approach
                fr_topk_norms = torch.zeros(fq_c.shape[0], K, device=fq_c.device)
                gs = 20
                for g in range(0, K, gs):
                    g_end = min(g + gs, K)
                    idx_g = s_topk_idx[:, g:g_end]
                    fr_g = f_r.index_select(0, idx_g.reshape(-1)).reshape(fq_c.shape[0], -1, fq_c.shape[1])
                    fr_topk_norms[:, g:g_end] = fr_g.norm(dim=2)
                num = 2.0 * fq_norm * fr_topk_norms
                den = (fq_norm ** 2 + fr_topk_norms ** 2).clamp_min(cfg.eps)
                alpha_topk = num / den

                w_topk = s_topk_vals * alpha_topk * mask.gather(1, s_topk_idx).to(s.dtype)

                # Sparse solve (grouped bmm, avoids full w @ f_r)
                f_tilde = sparse_solve(fq_c, f_r, s_topk_idx, w_topk,
                                       lam=cfg.lam, eps=cfg.eps)
            else:
                # === Original path: full matrices ===
                s, mask, anchor_idx, tau = retrieve(fq_c, f_r, rule=cfg.prefix_rule)
                alpha = norm_compatibility(fq_c, f_r, eps=cfg.eps)
                w = edge_weights(s, mask, alpha, clamp_negative=cfg.clamp_negative_weights)
                f_tilde = solve_closed_form(fq_c, f_r, w, lam=cfg.lam, eps=cfg.eps)

            energy = patch_energy(fq_c, f_tilde, metric=cfg.score_metric, eps=cfg.eps)
            energies.append(energy)
            f_tildes.append(f_tilde)
            anchors.append(anchor_idx)
            taus.append(tau)
            nneigh.append(mask.sum(dim=1))

        energy = torch.cat(energies, dim=0)
        score = image_score(energy, reduction=cfg.image_reduction)
        amap = None
        if grid_hw is not None:
            amap = energy_to_map(energy, grid_hw, out_hw, cfg.gaussian_kernel, cfg.gaussian_sigma)
        return {
            "score": score,
            "patch_energy": energy,
            "anomaly_map": amap,
            "f_tilde": torch.cat(f_tildes, dim=0),
            "w": None,
            "neighbor_mask": None,
            "sim": None,
            "anchor_idx": torch.cat(anchors, dim=0),
            "tau": torch.cat(taus, dim=0),
            "num_neighbors": torch.cat(nneigh, dim=0),
        }

    # ------------------------------------------------------------------ images
    def _get_extractor(self):
        if self._extractor is None:
            from .features.dinov3 import DINOv3Extractor

            self._extractor = DINOv3Extractor(self.cfg)
        return self._extractor

    @torch.no_grad()
    def run_image(
        self,
        query_image,
        ref_images: Union[object, Sequence[object]],
    ) -> Dict[str, torch.Tensor]:
        """Run ANoCo on raw images (paths / PIL / tensors) using DINOv3-L/16 features."""
        ext = self._get_extractor()
        f_q, grid = ext.extract(query_image)
        if not isinstance(ref_images, (list, tuple)):
            ref_images = [ref_images]
        feats = [ext.extract(r)[0] for r in ref_images]
        f_r = torch.cat(feats, dim=0)
        out = self.score_features(
            f_q, f_r, grid_hw=grid, out_hw=(self.cfg.img_size, self.cfg.img_size)
        )
        out["grid_hw"] = grid
        return out
