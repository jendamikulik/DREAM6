#!/usr/bin/env python3
# ================================================================
# CMB Cold Spot — matched-filter detector (no healpy), v8 (robust)
# - Reads Planck HEALPix map(s) via astropy.io.fits
# - Samples a tangent-plane patch using astropy-healpix (bilinear)
# - Inverts β-profile for a "cold" template (negative blob)
# - ZNCC matched filter + 1e3 random-null draws (excludes CS vicinity)
# - Plots: patch+template contours, null histogram with z_obs,
#          and per-band matched amplitudes if multiple FITS are given
#
# Requirements: numpy, scipy, matplotlib, astropy, astropy-healpix
#
# Examples:
#   python cmb_coldspot_matched_filter.py \
#     --fits COM_CMB_IQU-commander_1024_R3.00_full.fits \
#     --center_l 209 --center_b -57 --patch_deg 30 --pixsize_arcmin 3
#
#   # multi-band (prints per-band amplitudes and SED plot)
#   python cmb_coldspot_matched_filter.py \
#     --fits HFI_SkyMap_100_2048_R3.01_full.fits HFI_SkyMap_143_2048_R3.01_full.fits HFI_SkyMap_217_2048_R3.01_full.fits \
#     --center_l 209 --center_b -57 --patch_deg 30 --pixsize_arcmin 3
# ================================================================

"""
python .\cmb_patch_matched_filter_no_healpy.py --fits COM_CMB_IQU-commander_1024_R2.02_full.fits --center_l 209 --center_b -57 --patch_deg 30 --pixsize_arcmin 3 --theta_c_deg 5 --theta_max_deg 15 --save_prefix coldspot_41.9sigma
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft as spfft
from scipy.stats import norm

from astropy.io import fits
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.coordinates import SkyCoord

# ---------------------------
# FITS / HEALPix utilities
# ---------------------------
def read_healpix_temperature(path):
    """
    Return (map_1d, nside, ordering, coordsys) from a Planck-like FITS.
    Tries BINTABLE (I/I_STOKES/TEMPERATURE) or 1D image vector.
    """
    data = None
    hdr = None
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, 'data', None) is None:
                continue
            # BINTABLE with Stokes columns?
            if hasattr(hdu, 'columns'):
                cols = [c.name.upper() for c in hdu.columns]
                for cand in ('I_STOKES', 'TEMPERATURE', 'I', 'T'):
                    if cand in cols:
                        data = np.asarray(hdu.data[cand], dtype=np.float64)
                        hdr = hdu.header
                        break
            # Image HDU that is a 1D HEALPix vector?
            if data is None and isinstance(hdu.data, np.ndarray):
                arr = np.array(hdu.data).squeeze()
                if arr.ndim == 1 and arr.size >= 12:
                    data = arr.astype(np.float64)
                    hdr = hdu.header
            if data is not None:
                break

        if data is None:
            raise RuntimeError(f"Could not find a 1D HEALPix temperature vector in: {path}")

        nside = int(hdr.get('NSIDE', 0)) or int(np.sqrt(data.size / 12))
        ordering = str(hdr.get('ORDERING', 'RING')).upper()
        coordsys = str(hdr.get('COORDSYS', hdr.get('COORDTYPE', 'G'))).upper()

    return data, nside, ordering, coordsys


def build_sampler(nside, ordering, coordsys='G'):
    order = 'ring' if ordering.startswith('RING') else 'nested'
    frame = 'galactic' if coordsys.upper().startswith('G') else 'icrs'
    return HEALPix(nside=nside, order=order, frame=frame)


def bilinear_sample(hp: HEALPix, data, lon_deg, lat_deg):
    lon = np.asarray(lon_deg) * u.deg
    lat = np.asarray(lat_deg) * u.deg
    # Try both APIs (version differences across astropy-healpix):
    try:
        sc = SkyCoord(lon, lat, frame=hp.frame)
        return hp.interpolate_bilinear_skycoord(sc, data)
    except Exception:
        try:
            return hp.interpolate_bilinear_lonlat(lon, lat, data)
        except Exception:
            ipix = hp.lonlat_to_healpix(lon, lat)
            return data[ipix]


# ---------------------------
# Patch & template
# ---------------------------
def make_tangent_patch(hp, m, l0_deg, b0_deg, patch_deg=30.0, pixsize_arcmin=3.0, interp='bilinear'):
    npix = int(np.round((patch_deg * 60.0) / pixsize_arcmin))
    npix = max(32, npix)
    half = patch_deg / 2.0
    x = np.linspace(-half, +half, npix)        # Δl (deg, tangent)
    y = np.linspace(-half, +half, npix)        # Δb (deg, tangent)
    X, Y = np.meshgrid(x, y, indexing='xy')

    b0 = np.deg2rad(b0_deg)
    lon = (l0_deg + X / np.cos(b0)) % 360.0
    lat = np.clip(b0_deg + Y, -90.0, 90.0)

    if interp == 'bilinear':
        vals = bilinear_sample(hp, m, lon, lat)
    else:
        ip = hp.lonlat_to_healpix(lon * u.deg, lat * u.deg)
        vals = m[ip]

    return vals.reshape((npix, npix)), X, Y


def beta_profile_negative(X_deg, Y_deg, theta_c_deg=5.0, theta_max_deg=15.0, beta=1.0):
    R = np.sqrt(X_deg**2 + Y_deg**2)
    x = np.deg2rad(R) / np.deg2rad(theta_c_deg)
    prof = 1.0 / (1.0 + x**2)**(3*beta - 0.5)
    prof[R > theta_max_deg] = 0.0
    m = prof.max()
    if m > 0:
        prof = prof / m
    return -prof  # cold (negative) template


# ---------------------------
# Matched filter (ZNCC)
# ---------------------------
def zncc_center_response(patch, templ):
    P = patch - np.nanmean(patch)
    T = templ - np.nanmean(templ)
    Pstd = np.nanstd(P); Tstd = np.nanstd(T)
    if Pstd == 0 or Tstd == 0:
        return 0.0, np.zeros_like(P)

    P /= Pstd; T /= Tstd
    Fp = spfft.rfftn(P)
    Ft = spfft.rfftn(T)
    corr = spfft.irfftn(np.conj(Ft) * Fp, s=P.shape)
    corr = np.fft.fftshift(corr)
    cy, cx = np.array(corr.shape)//2
    # Normalize by N to keep response in [-1,1] for white fields
    return corr[cy, cx]/patch.size, corr


# ---------------------------
# Depth metric (core - ring)
# ---------------------------
def core_ring_depth(patch, theta_c_deg, pix_arcmin):
    ny, nx = patch.shape
    cy, cx = ny//2, nx//2
    r_pix = int((theta_c_deg*60.0/pix_arcmin)/2)
    yy, xx = np.ogrid[-cy:ny-cy, -cx:nx-cx]
    rr2 = xx*xx + yy*yy
    core = rr2 <= r_pix*r_pix
    ring = (rr2 > r_pix*r_pix) & (rr2 <= (3*r_pix)**2)
    return float(np.nanmean(patch[core]) - np.nanmean(patch[ring]))


def outside_coldspot(l, b, cs_l=209.0, cs_b=-57.0, exc_deg=10.0):
    p = SkyCoord(l*u.deg, b*u.deg, frame='galactic')
    c = SkyCoord(cs_l*u.deg, cs_b*u.deg, frame='galactic')
    return p.separation(c).deg >= exc_deg


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", nargs="+", required=True, help="One or more Planck HEALPix FITS files")
    ap.add_argument("--center_l", type=float, default=209.0)
    ap.add_argument("--center_b", type=float, default=-57.0)
    ap.add_argument("--patch_deg", type=float, default=30.0)
    ap.add_argument("--pixsize_arcmin", type=float, default=3.0)
    ap.add_argument("--theta_c_deg", type=float, default=5.0)
    ap.add_argument("--theta_max_deg", type=float, default=15.0)
    ap.add_argument("--interp", choices=["bilinear","nearest"], default="bilinear")
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_prefix", default="coldspot_v8")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load all maps and create samplers
    maps = []
    samplers = []
    for path in args.fits:
        m, nside, ordering, coordsys = read_healpix_temperature(path)
        hp = build_sampler(nside, ordering, coordsys)
        maps.append(m); samplers.append(hp)
        print(f"[OK] {path} → NSIDE={nside} ORDER={ordering} COORD={coordsys}")

    # Work band-by-band (and collect per-band amplitudes if multiple FITS)
    Ahat_list = []
    freq_labels = []
    for idx, (m, hp) in enumerate(zip(maps, samplers)):
        # Try to guess a band label from filename
        lab = ""
        for token in ("100","143","217","353","545","857"):
            if token in args.fits[idx]:
                lab = f"{token} GHz"; break
        freq_labels.append(lab if lab else f"map{idx+1}")

        patch, X, Y = make_tangent_patch(hp, m,
                                         args.center_l, args.center_b,
                                         patch_deg=args.patch_deg,
                                         pixsize_arcmin=args.pixsize_arcmin,
                                         interp=args.interp)

        templ = beta_profile_negative(X, Y, args.theta_c_deg, args.theta_max_deg)
        resp, corr = zncc_center_response(patch, templ)
        depth = core_ring_depth(patch, args.theta_c_deg, args.pixsize_arcmin)
        Ahat_list.append(resp)

        # ---------- Random nulls (uniform on sky, |b|>20°, exclude CS vicinity) ----------
        null_vals = []
        count = 0
        while count < args.n_null:
            l = rng.uniform(0, 360)
            b = rng.uniform(-90, 90)
            if abs(b) < 20 or not outside_coldspot(l, b):
                continue
            p_rand, XR, YR = make_tangent_patch(hp, m, l, b,
                                                patch_deg=args.patch_deg,
                                                pixsize_arcmin=args.pixsize_arcmin,
                                                interp=args.interp)
            t_rand = beta_profile_negative(XR, YR, args.theta_c_deg, args.theta_max_deg)
            r_rand, _ = zncc_center_response(p_rand, t_rand)
            null_vals.append(r_rand)
            count += 1

        null_vals = np.asarray(null_vals)
        mu, sig = float(null_vals.mean()), float(null_vals.std(ddof=1) or 1.0)
        z_obs = (resp - mu) / sig
        # small-tail p (two-sided as default)
        p_two = 2*min(norm.sf(abs(z_obs)), 1.0)

        print(f"\n=== {freq_labels[-1]} ===")
        print(f"Matched filter response (ZNCC): {resp:.3f}")
        print(f"Core–ring depth ΔT: {depth:.1f} μK")
        print(f"Nulls: μ={mu:.3f}, σ={sig:.3f} → z_obs={z_obs:.2f}, p≈{p_two:.3g}")

        # ---------- Figure 1: patch + template contours ----------
        fig1 = plt.figure(figsize=(6.4, 5.8))
        im = plt.imshow(patch, origin='lower',
                        extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap='viridis')
        cs = plt.contour(X, Y, -templ, levels=8, colors='k', linewidths=0.7, alpha=0.6)
        plt.clabel(cs, fmt=" ", fontsize=6)
        plt.xlabel("Δl (deg)"); plt.ylabel("Δb (deg)")
        plt.title(f"Cold Spot — patch @ {freq_labels[-1]}")
        cbar = plt.colorbar(im); cbar.set_label("K (arb.)")
        fig1.tight_layout()
        fig1.savefig(f"{args.save_prefix}_{idx+1}_patch.png", dpi=140)
        plt.close(fig1)

        # ---------- Figure 2: histogram of nulls with observed line ----------
        fig2 = plt.figure(figsize=(6.8, 4.2))
        plt.hist(null_vals, bins=48, density=True, alpha=0.65)
        plt.axvline(resp, color='tab:orange', lw=2, ls='--',
                    label=f"z={z_obs:.2f}, p≈{p_two:.3g}")
        plt.xlabel("ZNCC"); plt.ylabel("PDF")
        plt.title(f"Matched-filter nulls ({freq_labels[-1]})")
        plt.legend()
        fig2.tight_layout()
        fig2.savefig(f"{args.save_prefix}_{idx+1}_nulls.png", dpi=140)
        plt.close(fig2)

    # ---------- Figure 3: per-band amplitudes (if multiple maps) ----------
    if len(Ahat_list) > 1:
        freqs = []
        for lab in freq_labels:
            try:
                freqs.append(int(lab.split()[0]))
            except Exception:
                freqs.append(None)
        fig3 = plt.figure(figsize=(6.6, 4.2))
        x = np.arange(len(Ahat_list))
        plt.plot(x, Ahat_list, 'o-', lw=1.8)
        plt.xticks(x, freq_labels, rotation=0)
        plt.xlabel("Band"); plt.ylabel("Matched amplitude Â (arb.)")
        plt.title("Per-band matched amplitude (template-consistent?)")
        fig3.tight_layout()
        fig3.savefig(f"{args.save_prefix}_perband.png", dpi=150)
        plt.close(fig3)

    print("\nSaved figures:")
    print(f"  {args.save_prefix}_*_patch.png")
    print(f"  {args.save_prefix}_*_nulls.png")
    if len(Ahat_list) > 1:
        print(f"  {args.save_prefix}_perband.png")


if __name__ == "__main__":
    main()
