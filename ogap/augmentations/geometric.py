"""Channel-aware geometric augmentation for the v9.1 training path.

Why this module exists
----------------------
The legacy ``MRIPhysicsAugmentor`` applied geometric warps (flip + elastic) to the
``(x, y)`` pair. When the v9.1 physics engine is enabled it *replaces* the legacy
augmentor entirely — but the v9.1 :class:`~ogap.augmentations.physics_augment.PhysicsAugmentor`
is intensity/noise/k-space only, with **no** spatial transforms. The v9.1 stack
therefore trained with zero flips/affine/elastic, which is an unforced accuracy
loss for 3D tumour segmentation. This module restores spatial augmentation for the
v9.1 path.

Channel-role contract (spec §10)
--------------------------------
Geometric transforms are **label-coupled** and must be applied to *every* channel
plus the label so MRI, appended FSL-FAST tissue priors, and the segmentation mask
stay co-registered:

* **flip / affine / elastic** → all channels of ``x`` + the label ``y``.
* image channels are sampled with linear interpolation; the label with **nearest**.

This is the complement of the intensity shield in ``physics_augment.py`` (which
protects the *trailing tissue-prior* channels from intensity transforms only).

Shapes
------
Every callable accepts either a single ``(C, D, H, W)`` sample (CPU dataset path)
or a batched ``(B, C, D, H, W)`` tensor on any device (GPU prefetcher path), with
an optional integer label ``(D, H, W)`` / ``(B, D, H, W)``. Per-sample random
parameters are drawn independently, matching the CPU per-sample behaviour.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _cfg_get(config: Any, path: Sequence[str], default: Any) -> Any:
    node = config
    for key in path:
        if node is None:
            return default
        node = node.get(key, None) if isinstance(node, dict) else getattr(node, key, None)
    return default if node is None else node


def _cpu_generator(g: Optional[torch.Generator]) -> Optional[torch.Generator]:
    """Return ``g`` only if it is a CPU generator (scalar param draws are on CPU)."""
    if g is None:
        return None
    try:
        return g if torch.device(getattr(g, "device", "cpu")).type == "cpu" else None
    except (TypeError, RuntimeError):
        return None


def _u(lo: float, hi: float, g: Optional[torch.Generator]) -> float:
    return float(torch.empty(1).uniform_(lo, hi, generator=_cpu_generator(g)).item())


def _as_batched(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return ``(x5d, was_4d)`` adding a leading batch dim to a ``(C,D,H,W)`` input."""
    if x.dim() == 5:
        return x, False
    if x.dim() == 4:
        return x.unsqueeze(0), True
    raise ValueError(f"geometric augmentation expects 4D or 5D input, got {tuple(x.shape)}")


def random_flip(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    axes: Sequence[int] = (0, 1, 2),
    p: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Independent per-spatial-axis flip (Bernoulli ``p``) of all channels + label.

    Spatial axes index ``(D, H, W)`` as 0/1/2. The same flip is applied to ``x`` and
    ``y`` so they stay co-registered. Per-sample independent for a batched input.
    """
    xb, was_4d = _as_batched(x)
    yb = y.unsqueeze(0) if (y is not None and y.dim() == 3) else y
    out = xb.clone()
    out_y = yb.clone() if yb is not None else None
    for b in range(xb.shape[0]):
        flip_dims = []
        for ax in axes:
            if _u(0.0, 1.0, generator) < p:
                # xb[b] is (C, D, H, W): spatial axis ax lives at dim ax + 1.
                flip_dims.append(ax + 1)
        if flip_dims:
            out[b] = torch.flip(xb[b], dims=flip_dims)
            if out_y is not None:
                # yb[b] is (D, H, W): same spatial axis is at dim ax (one less).
                y_dims = [d - 1 for d in flip_dims]
                out_y[b] = torch.flip(yb[b], dims=y_dims)
    if was_4d:
        out = out.squeeze(0)
        if out_y is not None and y is not None and y.dim() == 3:
            out_y = out_y.squeeze(0)
    return out, out_y


def _affine_theta(
    rot_deg: float, scale: float, shear: float, axis_order: Tuple[int, int, int],
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build a (3, 4) affine matrix: rotation about one axis + isotropic scale + shear."""
    a = torch.tensor(rot_deg * 3.141592653589793 / 180.0, device=device, dtype=dtype)
    ca, sa = torch.cos(a), torch.sin(a)
    s = torch.tensor(float(scale), device=device, dtype=dtype)
    sh = torch.tensor(float(shear), device=device, dtype=dtype)
    eye = torch.eye(3, device=device, dtype=dtype)
    # Rotation in the plane orthogonal to axis_order[0].
    i, j, k = axis_order
    rot = eye.clone()
    rot[j, j] = ca; rot[j, k] = -sa
    rot[k, j] = sa; rot[k, k] = ca
    # Light shear coupling j<-k keeps it a small generic affine, not a pure rotation.
    rot[j, k] = rot[j, k] + sh
    mat = rot * s
    theta = torch.zeros(3, 4, device=device, dtype=dtype)
    theta[:, :3] = mat
    return theta


def random_affine(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    max_rotation_deg: float = 10.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    max_shear: float = 0.05,
    p: float = 0.3,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Small random affine (rotation + scale + shear) via ``grid_sample``.

    Image channels use trilinear sampling; the label uses nearest so it stays a
    valid integer mask. ``align_corners=False`` and zero padding match the
    skull-stripped zero-background convention.
    """
    xb, was_4d = _as_batched(x)
    yb = y.unsqueeze(0) if (y is not None and y.dim() == 3) else y
    b = xb.shape[0]
    device, dtype = xb.device, torch.float32
    thetas = []
    apply_mask = []
    for _ in range(b):
        if _u(0.0, 1.0, generator) >= p:
            thetas.append(torch.cat([torch.eye(3, device=device, dtype=dtype),
                                     torch.zeros(3, 1, device=device, dtype=dtype)], dim=1))
            apply_mask.append(False)
            continue
        rot = _u(-max_rotation_deg, max_rotation_deg, generator)
        sc = _u(scale_range[0], scale_range[1], generator)
        sh = _u(-max_shear, max_shear, generator)
        # Rotate about a randomly chosen principal axis.
        axis = int(torch.randint(0, 3, (1,), generator=_cpu_generator(generator)).item())
        order = ([0, 1, 2][axis:] + [0, 1, 2][:axis])
        thetas.append(_affine_theta(rot, sc, sh, tuple(order), device, dtype))
        apply_mask.append(True)
    if not any(apply_mask):
        if was_4d:
            return xb.squeeze(0), (yb.squeeze(0) if (yb is not None and y is not None and y.dim() == 3) else y)
        return xb, y
    theta = torch.stack(thetas, dim=0)  # (B, 3, 4)
    grid = F.affine_grid(theta, list(xb.shape), align_corners=False)
    x_warp = F.grid_sample(xb.float(), grid, mode="bilinear",
                           padding_mode="zeros", align_corners=False).to(xb.dtype)
    # Only overwrite the samples we actually transformed (leave the rest bit-exact).
    m = torch.tensor(apply_mask, device=device)
    out = torch.where(m.view(-1, 1, 1, 1, 1), x_warp, xb)
    out_y = yb
    if yb is not None:
        y_warp = F.grid_sample(yb.unsqueeze(1).float(), grid, mode="nearest",
                               padding_mode="zeros", align_corners=False).squeeze(1).to(yb.dtype)
        out_y = torch.where(m.view(-1, 1, 1, 1), y_warp, yb)
    if was_4d:
        out = out.squeeze(0)
        if out_y is not None and y is not None and y.dim() == 3:
            out_y = out_y.squeeze(0)
    return out, out_y


def random_elastic(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    alpha: float = 8.0,
    sigma_grid: int = 4,
    p: float = 0.2,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Smooth random elastic deformation (low-res displacement field, trilinearly
    upsampled), applied identically to all channels + label.

    ``alpha`` is the displacement magnitude in voxels; ``sigma_grid`` is the
    control-grid resolution (larger = finer/rougher field). Label uses nearest.
    """
    xb, was_4d = _as_batched(x)
    yb = y.unsqueeze(0) if (y is not None and y.dim() == 3) else y
    b, _, d, h, w = xb.shape
    device = xb.device
    base = F.affine_grid(
        torch.eye(3, 4, device=device, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1),
        list(xb.shape), align_corners=False,
    )  # (B, D, H, W, 3) in [-1, 1]
    out = xb.clone()
    out_y = yb.clone() if yb is not None else None
    g_cpu = _cpu_generator(generator)
    for bi in range(b):
        if _u(0.0, 1.0, generator) >= p:
            continue
        low = torch.randn(1, 3, sigma_grid, sigma_grid, sigma_grid,
                          generator=g_cpu, device="cpu").to(device)
        disp = F.interpolate(low, size=(d, h, w), mode="trilinear", align_corners=False)
        # Normalise voxel magnitude to grid coords (range 2 over the axis length).
        disp = disp.squeeze(0).permute(1, 2, 3, 0)  # (D, H, W, 3)
        scale = torch.tensor([2.0 * alpha / max(w, 1), 2.0 * alpha / max(h, 1),
                              2.0 * alpha / max(d, 1)], device=device)
        grid = (base[bi] + disp * scale).unsqueeze(0)
        out[bi] = F.grid_sample(xb[bi:bi + 1].float(), grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False).squeeze(0).to(xb.dtype)
        if out_y is not None:
            out_y[bi] = F.grid_sample(yb[bi:bi + 1].unsqueeze(1).float(), grid, mode="nearest",
                                      padding_mode="zeros", align_corners=False).squeeze(1).squeeze(0).to(yb.dtype)
    if was_4d:
        out = out.squeeze(0)
        if out_y is not None and y is not None and y.dim() == 3:
            out_y = out_y.squeeze(0)
    return out, out_y


class GeometricAugmentor:
    """Composed, config-driven geometric augmentation over ``(x, y)``.

    Applied to ALL channels of ``x`` plus the label, so MRI + appended tissue
    priors + mask stay co-registered. Reads ``augmentation.geometric.*``.
    """

    def __init__(self, config: Any) -> None:
        g = ("augmentation", "geometric")
        self.enabled = bool(_cfg_get(config, g + ("enabled",), False))
        self.do_flip = bool(_cfg_get(config, g + ("flip",), True))
        self.flip_p = float(_cfg_get(config, g + ("flip_p",), 0.5))
        self.do_affine = bool(_cfg_get(config, g + ("affine",), True))
        self.affine_p = float(_cfg_get(config, g + ("affine_p",), 0.3))
        self.max_rotation_deg = float(_cfg_get(config, g + ("max_rotation_deg",), 10.0))
        self.scale_range = tuple(_cfg_get(config, g + ("scale_range",), (0.9, 1.1)))
        self.max_shear = float(_cfg_get(config, g + ("max_shear",), 0.05))
        self.do_elastic = bool(_cfg_get(config, g + ("elastic",), False))
        self.elastic_p = float(_cfg_get(config, g + ("elastic_p",), 0.2))
        self.elastic_alpha = float(_cfg_get(config, g + ("elastic_alpha",), 8.0))
        self.elastic_grid = int(_cfg_get(config, g + ("elastic_grid",), 4))

    def has_work(self) -> bool:
        return self.enabled and (self.do_flip or self.do_affine or self.do_elastic)

    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                 generator: Optional[torch.Generator] = None
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.has_work():
            return x, y
        if self.do_flip:
            x, y = random_flip(x, y, p=self.flip_p, generator=generator)
        if self.do_affine:
            x, y = random_affine(x, y, max_rotation_deg=self.max_rotation_deg,
                                 scale_range=self.scale_range, max_shear=self.max_shear,
                                 p=self.affine_p, generator=generator)
        if self.do_elastic:
            x, y = random_elastic(x, y, alpha=self.elastic_alpha,
                                  sigma_grid=self.elastic_grid, p=self.elastic_p,
                                  generator=generator)
        return x, y


def build_geometric_augmentor(config: Any) -> Optional["GeometricAugmentor"]:
    """Return a :class:`GeometricAugmentor`, or ``None`` when nothing is enabled."""
    aug = GeometricAugmentor(config)
    return aug if aug.has_work() else None
