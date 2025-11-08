#!/usr/bin/env python3
# === CMB PATCH MATCHED FILTER (NO healpy) ===
# Detect circular signatures (Cold Spot, SZ, textures) using a flat-sky patch
# extracted from a HEALPix map, without using healpy.
#
# Requirements:
#   pip install numpy scipy matplotlib astropy astropy-healpix
#
# Usage:
#   python cmb_patch_matched_filter_no_healpy.py \
#       --fits /path/to/COM_CMB_IQU-commander_1024_R2.02_full.fits \
#       --center_l -55 --center_b -30 \
#       --patch_deg 20 --pixsize_arcmin 5 \
#       --theta_c_deg 5 --theta_max_deg 15
#
# Notes:
# - This script reads the HEALPix map using astropy.io.fits and samples it
#   via astropy_healpix.HEALPix (bilinear if available, else nearest). No healpy is used.
# - It works in a *local flat-sky approximation* on a tangent-plane patch.
# - The matched filter is implemented as a whitened, normalized cross-correlation
#   (ZNCC) via FFTs. For many use-cases this is an effective proxy.

import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import fft as spfft

def read_healpix_temperature(path):
    """Load temperature vector and NSIDE/ORDERING from a Planck IQU FITS file."""
    with fits.open(path) as hdul:
        hdr = None
        data = None
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None:
                if hasattr(hdu, 'columns'):
                    cols = [c.name.upper() for c in hdu.columns]
                    for cand in ['I_STOKES','TEMPERATURE','I','T']:
                        if cand in cols:
                            data = hdu.data[cand].astype(np.float64)
                            hdr = hdu.header
                            break
                if data is None and isinstance(hdu.data, np.ndarray):
                    arr = np.array(hdu.data).squeeze()
                    if arr.ndim == 1 and arr.size > 12:
                        data = arr.astype(np.float64)
                        hdr = hdu.header
            if data is not None and hdr is not None:
                break
        if data is None or hdr is None:
            raise RuntimeError("Could not locate HEALPix temperature vector in FITS.")

        nside = int(hdr.get('NSIDE', 0))
        ordering = str(hdr.get('ORDERING', 'RING')).upper()
        coordsys = str(hdr.get('COORDSYS', hdr.get('COORDTYPE', 'G'))).upper()
        if nside <= 0:
            prim = hdul[0].header
            nside = int(prim.get('NSIDE', 0)) or nside
            ordering = str(prim.get('ORDERING', ordering)).upper()
            coordsys = str(prim.get('COORDSYS', coordsys)).upper()

    return data, nside, ordering, coordsys

def build_healpix_sampler(nside, ordering, frame='galactic'):
    """Return an astropy-healpix sampler object in the desired frame."""
    order = 'ring' if ordering.startswith('RING') else 'nested'
    if frame.lower().startswith('g'):
        frame = 'galactic'
    else:
        frame = 'icrs'
    hp = HEALPix(nside=nside, order=order, frame=frame)
    return hp

def sample_bilinear(hp, data, lon_deg, lat_deg):
    """
    Try several astropy-healpix interpolation APIs, falling back to nearest neighbor.
    lon_deg, lat_deg are numpy arrays (degrees).
    """
    lonq = lon_deg * u.deg
    latq = lat_deg * u.deg
    # 1) interpolate_bilinear_skycoord
    try:
        sc = SkyCoord(lonq, latq, frame=hp.frame)
        vals = hp.interpolate_bilinear_skycoord(sc, data)
        return vals
    except Exception:
        pass
    # 2) interpolate_bilinear_lonlat (older/newer alt API)
    try:
        vals = hp.interpolate_bilinear_lonlat(lonq, latq, data)
        return vals
    except Exception:
        pass
    # 3) Fallback to nearest
    try:
        ipix = hp.lonlat_to_healpix(lonq, latq)
        return data[ipix]
    except Exception as e:
        raise RuntimeError(f"Sampling failed with astropy-healpix: {e}")

def make_tangent_patch(hp, map_values, center_l_deg, center_b_deg,
                       patch_deg=20.0, pixsize_arcmin=5.0, interp='bilinear'):
    """
    Sample a tangent-plane patch of size patch_deg x patch_deg (degrees) around (l,b).
    Returns patch (2D), and coordinate grids (x_deg, y_deg) where x~lon, y~lat offsets.
    """
    npix = int(np.round((patch_deg * 60.0) / pixsize_arcmin))
    npix = max(32, npix)
    size = patch_deg

    half = size / 2.0
    x = np.linspace(-half, half, npix)  # deg
    y = np.linspace(-half, half, npix)  # deg
    X, Y = np.meshgrid(x, y, indexing='xy')

    b0 = np.deg2rad(center_b_deg)
    lon = center_l_deg + X / np.cos(b0)
    lat = center_b_deg + Y

    lon = (lon + 360.0) % 360.0
    lat = np.clip(lat, -90.0, 90.0)

    if interp == 'bilinear':
        vals = sample_bilinear(hp, map_values, lon, lat)
    else:
        ipix = hp.lonlat_to_healpix(lon * u.deg, lat * u.deg)
        vals = map_values[ipix]

    patch = vals.reshape((npix, npix))
    return patch, X, Y

def beta_profile_2d(X_deg, Y_deg, theta_c_deg=5.0, theta_max_deg=15.0, T0=1.0, beta=1.0):
    """β-profile on a flat patch (θ in degrees)."""
    R = np.sqrt(X_deg**2 + Y_deg**2)
    theta = np.deg2rad(R)
    theta_c = np.deg2rad(theta_c_deg)
    x = theta / theta_c
    prof = T0 / (1 + x**2)**(3*beta - 0.5)
    prof[R > theta_max_deg] = 0.0
    m = prof.max()
    if m > 0:
        prof = prof / m
    return prof

def zncc_fft_match(patch, template):
    """Zero-mean normalized cross-correlation via FFT; return center response and full map."""
    P = patch - np.nanmean(patch)
    T = template - np.nanmean(template)
    Pstd = np.nanstd(P)
    Tstd = np.nanstd(T)
    if Pstd == 0 or Tstd == 0:
        raise RuntimeError("Zero variance in patch or template.")
    P /= Pstd
    T /= Tstd

    Fp = spfft.rfftn(P)
    Ft = spfft.rfftn(T)
    corr = spfft.irfftn(np.conj(Ft) * Fp, s=P.shape)
    corr = np.fft.fftshift(corr)
    cy, cx = corr.shape[0]//2, corr.shape[1]//2
    response = corr[cy, cx]
    return response, corr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="Path to Planck IQU HEALPix FITS (Commander/SMICA)")
    ap.add_argument("--center_l", type=float, default=-55.0, help="Center longitude (deg) in Galactic")
    ap.add_argument("--center_b", type=float, default=-30.0, help="Center latitude (deg) in Galactic")
    ap.add_argument("--patch_deg", type=float, default=20.0, help="Patch size (deg)")
    ap.add_argument("--pixsize_arcmin", type=float, default=5.0, help="Pixel size (arcmin)")
    ap.add_argument("--theta_c_deg", type=float, default=5.0, help="β-profile core radius (deg)")
    ap.add_argument("--theta_max_deg", type=float, default=15.0, help="β-profile truncation radius (deg)")
    ap.add_argument("--interp", choices=["nearest","bilinear"], default="bilinear", help="Sampling interpolation")
    ap.add_argument("--save_prefix", type=str, default="coldspot_patch", help="Output filename prefix")
    args = ap.parse_args()

    m, nside, ordering, coordsys = read_healpix_temperature(args.fits)
    print(f"Loaded HEALPix map: NSIDE={nside}, ORDERING={ordering}, COORDSYS={coordsys}")

    hp = build_healpix_sampler(nside, ordering, frame='galactic')

    patch, X, Y = make_tangent_patch(hp, m, args.center_l, args.center_b,
                                     patch_deg=args.patch_deg,
                                     pixsize_arcmin=args.pixsize_arcmin,
                                     interp=args.interp)

    templ = beta_profile_2d(X, Y, theta_c_deg=args.theta_c_deg, theta_max_deg=args.theta_max_deg)

    response, corrmap = zncc_fft_match(patch, templ)
    print(f"Matched-filter (ZNCC) response at center: {response:.3f}")

    fig1 = plt.figure(figsize=(6,5))
    plt.imshow(patch, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel("Δl (deg)")
    plt.ylabel("Δb (deg)")
    plt.title("CMB Patch (μK)")
    plt.colorbar(label="μK")
    fig1.tight_layout()
    fig1.savefig(f"{args.save_prefix}_map.png", dpi=150)

    fig2 = plt.figure(figsize=(6,5))
    plt.imshow(templ, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel("Δl (deg)")
    plt.ylabel("Δb (deg)")
    plt.title("β-profile Template (unit peak)")
    plt.colorbar()
    fig2.tight_layout()
    fig2.savefig(f"{args.save_prefix}_template.png", dpi=150)

    fig3 = plt.figure(figsize=(6,5))
    plt.imshow(corrmap, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel("Δl (deg)")
    plt.ylabel("Δb (deg)")
    plt.title("ZNCC Correlation Map")
    plt.colorbar()
    fig3.tight_layout()
    fig3.savefig(f"{args.save_prefix}_corr.png", dpi=150)

    print(f"Saved: {args.save_prefix}_map.png, {args.save_prefix}_template.png, {args.save_prefix}_corr.png")

if __name__ == "__main__":
    main()
