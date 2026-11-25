"""
Two-point photometric normalisation for longitudinal fish imaging
=================================================================

Pipeline
--------
    1. inverse sRGB transfer function        -> values proportional to scene radiance
    2. two-point affine correction            -> cancels per-image gain AND offset,
       using in-frame eye (black) and ID-tag (white) references, both measured
       in LINEAR light
    3. forward sRGB transfer function         -> 8-bit output for the segmentation models

Why this and not EXIF exposure normalisation
--------------------------------------------
For a camera response  L' = g*L + o  (gain g, additive offset o from veiling glare,
lens flare, black-level drift, ambient scatter):

    (L' - L'_black) / (L'_white - L'_black)
        = (gL + o - gLb - o) / (gLw + o - gLb - o)
        = (L - Lb) / (Lw - Lb)

Both g and o cancel exactly. Dividing by EXIF sensor exposure removes only g, and
leaves o untouched - and o is precisely what Weber contrast is vulnerable to, since
a multiplicative gain already cancels in any ratio.

NOTE ON THE PREVIOUS IMPLEMENTATION
-----------------------------------
The earlier formula  (1/alpha)*L - beta  divides before subtracting and therefore
cancels neither g nor o. It is replaced here. See `demo_invariance()`.

Usage
-----
    python photometric_normalization_two_point.py --group stressed
        (= --images data/stressed/images/head_images --labels data/stressed/labels/tags_eyes
           --rois data/stressed/images/operculum_images
           --out data/stressed/images/normalized_operculums_two_point)
    python photometric_normalization_two_point.py --images ... --labels ... --rois ... --out ...
or import and call `run_pipeline(...)`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """Encoded [0,1] -> linear (proportional to scene radiance)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    """Linear [0,1] -> encoded."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


LUMA = np.array([0.2126, 0.7152, 0.0722])


def linear_luminance(path: str | Path, size: int | None = 1080) -> np.ndarray:
    """Read an image and return linear-light luminance in [0,1]."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    rgb_lin = srgb_to_linear(np.asarray(img, dtype=np.float64) / 255.0)
    return rgb_lin @ LUMA


@dataclass
class References:
    """Per-image black and white anchors, measured in linear light."""
    key: str
    black: float
    white: float
    span: float = field(init=False)
    n_black: int = 0
    n_white: int = 0
    ok: bool = field(init=False)

    def __post_init__(self):
        self.span = self.white - self.black
        self.ok = (self.span > 1e-6) and (self.n_black > 0) and (self.n_white > 0)


def polygon_mask(vertices_norm: list[float], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a normalised polygon to a boolean mask."""
    h, w = shape
    pts = [
        (vertices_norm[i] * w, vertices_norm[i + 1] * h)
        for i in range(0, len(vertices_norm) - 1, 2)
    ]
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, outline=1, fill=1)
    return np.asarray(m, dtype=bool)


def extract_references(
    key: str,
    luminance: np.ndarray,
    labels: list[list[str]],
    eye_class: int = 0,
    pct: float = 0.20,
) -> References:
    """
    Measure the black and white anchors from in-frame reference regions.

    Parameters
    ----------
    luminance : LINEAR-light luminance image (this is the essential difference
                from the earlier CIELAB implementation).
    pct       : fraction of extreme pixels averaged within each region. Averaging
                a tail rather than taking the single extreme pixel makes the
                anchors robust to specular highlights on wet tissue and to
                isolated sensor noise.
    """
    eye_px, tag_px = [], []
    for row in labels:
        vals = [float(x) for x in row]
        cls, verts = int(vals[0]), vals[1:]
        mask = polygon_mask(verts, luminance.shape)
        px = luminance[mask]
        if px.size == 0:
            continue
        (eye_px if cls == eye_class else tag_px).append(px)

    if not eye_px or not tag_px:
        return References(key, np.nan, np.nan, n_black=0, n_white=0)

    eye = np.concatenate(eye_px)
    tag = np.concatenate(tag_px)

    k_e = max(1, int(pct * eye.size))
    k_t = max(1, int(pct * tag.size))

    black = np.sort(eye)[:k_e].mean()
    white = np.sort(tag)[::-1][:k_t].mean()

    return References(key, float(black), float(white), n_black=k_e, n_white=k_t)


def two_point_correct(luminance: np.ndarray, ref: References) -> np.ndarray:
    """
    (L - L_black) / (L_white - L_black), applied in linear light.

    Cancels any per-image affine camera response L' = g*L + o.
    Values outside [0,1] are retained (not clipped) so that the correction stays
    invertible; clipping happens only at the 8-bit re-encoding step.
    """
    if not ref.ok:
        raise ValueError(f"unusable references for {ref.key}: {ref}")
    return (luminance - ref.black) / ref.span


def exif_exposure_correct(luminance: np.ndarray, H: float, H_ref: float) -> np.ndarray:
    """EXIF exposure normalisation, retained for the comparison only."""
    return luminance * (H_ref / H)


def exposure_from_exif(shutter: float, iso: float, fnumber: float) -> float:
    """Relative sensor exposure  H = t * (ISO/100) / N^2."""
    return shutter * (iso / 100.0) / (fnumber ** 2)


def descriptors(lum: np.ndarray, spot_mask: np.ndarray, bg_mask: np.ndarray) -> dict:
    """
    All descriptors computed on LINEAR values, averaging pixels before any ratio.

    This ordering matters: eotf(mean) != mean(eotf), and the discrepancy is larger
    for the heterogeneous background annulus than for spot interiors.
    """
    s = float(lum[spot_mask].mean())
    b = float(lum[bg_mask].mean())
    return {
        "spot_lin": s,
        "bg_lin": b,
        "weber": (s - b) / b,
        "log_ratio": float(np.log(s / b)) if s > 0 and b > 0 else np.nan,
        "area_px": int(spot_mask.sum()),
    }


def load_labels(path: str | Path) -> list[list[str]]:
    with open(path) as fh:
        return [ln.split() for ln in fh if ln.strip()]


def run_pipeline(
    image_dir: str | Path,
    label_dir: str | Path,
    roi_dir: str | Path,
    out_dir: str | Path,
    exif_csv: str | Path | None = None,
    size: int = 1080,
    pct: float = 0.20,
    save_images: bool = True,
) -> pd.DataFrame:
    """
    Returns a per-image table of reference anchors and, if EXIF is supplied,
    the exposure factor, so that the two corrections can be compared.
    """
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    roi_dir, out_dir = Path(roi_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exif = None
    if exif_csv is not None:
        exif = pd.read_csv(exif_csv)
        exif["key"] = (exif.file.str.replace("\\", "/", regex=False)
                               .str.split("/").str[-1]
                               .str.replace(".JPG", "", regex=False)
                               .str.replace(".jpg", "", regex=False))
        exif["H"] = exposure_from_exif(exif.shutter, exif.iso, exif.fnum)
        H_ref = float(np.median(exif.H))
        exif = exif.set_index("key")

    rows, failures = [], []

    for img_path in sorted(image_dir.glob("*.[jJ][pP][gG]")):
        key = img_path.stem
        label_path = label_dir / f"{key}.txt"
        roi_path = roi_dir / f"{key}.jpg"

        if not label_path.exists():
            failures.append((key, "missing label file")); continue
        if not roi_path.exists():
            failures.append((key, "missing ROI crop")); continue

        lum_full = linear_luminance(img_path, size=size)
        ref = extract_references(key, lum_full, load_labels(label_path), pct=pct)

        if not ref.ok:
            failures.append((key, f"bad references black={ref.black:.4g} white={ref.white:.4g}"))
            continue

        lum_roi = linear_luminance(roi_path, size=size)
        corrected = two_point_correct(lum_roi, ref)

        rec = dict(key=key, black=ref.black, white=ref.white, span=ref.span,
                   roi_mean_raw=float(lum_roi.mean()),
                   roi_mean_corrected=float(np.clip(corrected, 0, 1).mean()))

        if exif is not None and key in exif.index:
            H = float(exif.loc[key, "H"])
            rec["H"] = H
            rec["logH"] = float(np.log2(H))
            rec["roi_mean_exif"] = float(exif_exposure_correct(lum_roi, H, H_ref).mean())

        rows.append(rec)

        if save_images:
            out = (linear_to_srgb(np.clip(corrected, 0, 1)) * 255).round().astype(np.uint8)
            Image.fromarray(out).save(out_dir / f"{key}.jpg",
                                      quality=100, subsampling=0, optimize=True)

    df = pd.DataFrame(rows)

    if failures:
        print(f"\n{len(failures)} image(s) could not be processed:")
        for k, why in failures[:20]:
            print(f"   {k}: {why}")

    meta = {"transfer": "sRGB (IEC 61966-2-1)",
            "correction": "two-point affine on linear luminance",
            "reference_pct": pct, "size": size, "n_processed": len(df),
            "n_failed": len(failures)}
    (out_dir / "_normalization.json").write_text(json.dumps(meta, indent=2))

    return df


def reference_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    How stable are the anchors across sessions?

    Report this. Neither reference is a calibrated standard: the eye is not true
    black (non-zero pupil reflectance, specular highlights on wet tissue) and the
    ID tag is not a true white (wetting, viewing angle, soiling). The correction
    is only as trustworthy as the span between them.
    """
    out = df[["black", "white", "span"]].agg(["mean", "std", "min", "max"]).T
    out["CV_%"] = 100 * out["std"] / out["mean"]
    return out


def compare_corrections(df: pd.DataFrame, paired_key_fn=None) -> pd.DataFrame:
    """
    Side-by-side comparison of the two corrections on the same images.

    Expected pattern, which is the point of the comparison:
      - EXIF normalisation leaves Weber contrast and the log-ratio EXACTLY
        unchanged (both are invariant to a multiplicative gain), so it cannot
        be credited with any effect on the primary descriptor.
      - The two-point correction DOES change them, because it removes an
        additive offset, to which ratios are not invariant.
    """
    cols = [c for c in ("roi_mean_raw", "roi_mean_corrected", "roi_mean_exif")
            if c in df.columns]
    summary = df[cols].agg(["mean", "std"]).T
    if "logH" in df.columns:
        from scipy import stats
        rows = []
        for c in cols:
            r, p = stats.pearsonr(df.logH, df[c])
            rows.append({"measure": c, "r_with_logH": r, "p": p})
        summary = summary.join(pd.DataFrame(rows).set_index("measure"))
    return summary


def demo_invariance() -> None:
    """
    Shows why the two-point form is used and the earlier formula is not.
    Prints a worked example under a camera response L' = g*L + o.
    """
    L = np.array([0.05, 0.10, 0.30, 0.60])
    Lb, Lw = 0.02, 0.80
    g, o = 1.37, 0.05
    Lp, Lbp, Lwp = g * L + o, g * Lb + o, g * Lw + o

    old_corrupt = (1 / Lwp) * Lp - Lbp
    old_clean = (1 / Lw) * L - Lb
    new_corrupt = (Lp - Lbp) / (Lwp - Lbp)
    new_clean = (L - Lb) / (Lw - Lb)

    print("camera response: L' = %.2f*L + %.2f" % (g, o))
    print("  previous formula (1/alpha)*L - beta : invariant =",
          np.allclose(old_corrupt, old_clean))
    print("  two-point (L-Lb)/(Lw-Lb)            : invariant =",
          np.allclose(new_corrupt, new_clean))
    print("\n  Weber contrast (spot=L[0], background=L[3]):")
    for nm, (s, b) in [("uncorrected", (Lp[0], Lp[3])),
                       ("previous formula", (old_corrupt[0], old_corrupt[3])),
                       ("two-point", (new_corrupt[0], new_corrupt[3])),
                       ("ground truth", (L[0], L[3]))]:
        print("     %-18s %+.5f" % (nm, (s - b) / b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=("control", "stressed", "misc"), default=None,
                    help="fill --images/--labels/--rois/--out with the standard all_data/data/<group> paths")
    ap.add_argument("--images", help="full head images (eye + tag visible)")
    ap.add_argument("--labels", help="polygon labels, class 0 = eye")
    ap.add_argument("--rois", help="opercular ROI crops")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--exif", default=None, help="EXIF csv, for the comparison")
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--pct", type=float, default=0.20)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--demo", action="store_true", help="print the invariance demo and exit")
    a = ap.parse_args()

    if a.demo:
        demo_invariance(); return

    if a.group:
        g = Path(__file__).resolve().parents[2] / "data" / a.group
        a.images = a.images or str(g / "images" / "head_images")
        a.labels = a.labels or str(g / "labels" / "tags_eyes")
        a.rois = a.rois or str(g / "images" / "operculum_images")
        a.out = a.out or str(g / "images" / "normalized_operculums_two_point")

    missing = [f"--{n}" for n in ("images", "labels", "rois", "out") if getattr(a, n) is None]
    if missing:
        ap.error("missing required arguments: " + ", ".join(missing))

    df = run_pipeline(a.images, a.labels, a.rois, a.out, a.exif,
                      size=a.size, pct=a.pct, save_images=not a.no_save)
    df.to_csv(Path(a.out) / "reference_values.csv", index=False)

    print("\n=== reference anchor stability ===")
    print(reference_stability(df).round(5).to_string())
    print("\n=== correction comparison ===")
    print(compare_corrections(df).round(5).to_string())
    print(f"\nwrote {len(df)} images to {a.out}")


if __name__ == "__main__":
    main()
