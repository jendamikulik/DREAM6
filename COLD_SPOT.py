#!/usr/bin/env python3
# === CMB PATCH MATCHED FILTER (NO healpy) s KURTÓZOU ===
# Detekce a testování ne-Gaussovské statistiky (Kurtóza) Cold Spotu.
#
# Requirements:
#   pip install numpy scipy matplotlib astropy astropy-healpix

import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import fft as spfft
from scipy.stats import kurtosis as scipy_kurtosis  # NOVÝ IMPORT


# -------------------- HEALPix/Map Loading Functions --------------------

def read_healpix_temperature(path):
    """Load temperature vector and NSIDE/ORDERING from a Planck IQU FITS file."""
    with fits.open(path) as hdul:
        hdr = None
        data = None
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None:
                if hasattr(hdu, 'columns'):
                    cols = [c.name.upper() for c in hdu.columns]
                    for cand in ['I_STOKES', 'TEMPERATURE', 'I', 'T']:
                        if cand in cols:
                            # Používáme float64 pro přesnost
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
    """Try several astropy-healpix interpolation APIs, falling back to nearest neighbor."""
    lonq = lon_deg * u.deg
    latq = lat_deg * u.deg
    # 1) interpolate_bilinear_skycoord
    try:
        sc = SkyCoord(lonq, latq, frame=hp.frame)
        return hp.interpolate_bilinear_skycoord(sc, data)
    except Exception:
        pass
    # 2) interpolate_bilinear_lonlat (older/newer alt API)
    try:
        return hp.interpolate_bilinear_lonlat(lonq, latq, data)
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
    """Sample a tangent-plane patch."""
    npix = int(np.round((patch_deg * 60.0) / pixsize_arcmin))
    npix = max(32, npix)
    size = patch_deg

    half = size / 2.0
    x = np.linspace(-half, half, npix)  # deg
    y = np.linspace(-half, half, npix)  # deg
    X, Y = np.meshgrid(x, y, indexing='xy')

    b0 = np.deg2rad(center_b_deg)
    # Přesnější výpočet souřadnic pro tangenciální projekci
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


# -------------------- Matched Filter & Template Functions --------------------

def beta_profile_2d(X_deg, Y_deg, theta_c_deg=5.0, theta_max_deg=15.0, T0=1.0, beta=1.0):
    """β-profile on a flat patch (θ in degrees)."""
    R = np.sqrt(X_deg ** 2 + Y_deg ** 2)
    theta = np.deg2rad(R)
    theta_c = np.deg2rad(theta_c_deg)
    x = theta / theta_c
    # Použití β-profilu, který je často používán pro clustery/voidy (zde jako šablona)
    prof = T0 / (1 + x ** 2) ** (3 * beta - 0.5)
    prof[R > theta_max_deg] = 0.0
    m = prof.max()
    if m > 0:
        prof = prof / m
    return prof


def zncc_fft_match(patch, template):
    """Zero-mean normalized cross-correlation via FFT."""
    P = patch - np.nanmean(patch)
    T = template - np.nanmean(template)
    Pstd = np.nanstd(P)
    Tstd = np.nanstd(T)
    if Pstd == 0 or Tstd == 0:
        return np.nan, np.full_like(patch, np.nan)
    P /= Pstd
    T /= Tstd

    # Whitening se zde zjednodušuje na ZNCC (normalizovaná CC)
    Fp = spfft.rfftn(P)
    Ft = spfft.rfftn(T)
    corr = spfft.irfftn(np.conj(Ft) * Fp, s=P.shape)
    corr = np.fft.fftshift(corr)
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    response = corr[cy, cx]
    return response, corr


# -------------------- NOVÁ SEKCE: KURTÓZA --------------------

def local_kurtosis_analysis(patch, templ, R_core_mask=0.5, R_annulus_min=0.01, R_annulus_max=0.05):
    """
    Vypočítá Fisherovu (excess) kurtózu v centrální oblasti a pro kontrolu v okrajovém prstenci.

    R_core_mask: Template value threshold pro centrální masku (např. 0.5 = 50% max ampl.).
    R_annulus_min/max: Template values pro okrajový prstenec.
    """

    # 1. Odstranění průměru (stejně jako u Matched Filteru)
    P_mean_sub = patch - np.nanmean(patch)

    # 2. Vytvoření masek

    # Centrální oblast (silný signál anomálie)
    mask_center = (templ >= R_core_mask)

    # Okrajový prstenec (Čisté Gaussovské pozadí CMB pro srovnání)
    mask_annulus = (templ <= R_annulus_max) & (templ >= R_annulus_min)

    results = {}

    # --- Centrální oblast (Anomálie) ---
    vals_center = P_mean_sub[mask_center]
    if vals_center.size >= 100:
        # Fisherova kurtóza (excess kurtosis, odečteno -3), bez bias korekce
        K_center = scipy_kurtosis(vals_center, fisher=True, bias=False)
        results['K_center'] = K_center
        results['N_center'] = vals_center.size
    else:
        results['K_center'] = np.nan
        results['N_center'] = vals_center.size

    # --- Okrajový prstenec (Kontrola pozadí) ---
    vals_annulus = P_mean_sub[mask_annulus]
    if vals_annulus.size >= 100:
        K_annulus = scipy_kurtosis(vals_annulus, fisher=True, bias=False)
        results['K_annulus'] = K_annulus
        results['N_annulus'] = vals_annulus.size
    else:
        results['K_annulus'] = np.nan
        results['N_annulus'] = vals_annulus.size

    return results


# -------------------- MAIN --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="Path to Planck IQU HEALPix FITS (Commander/SMICA)")
    ap.add_argument("--center_l", type=float, default=-55.0, help="Center longitude (deg) in Galactic (l)")
    ap.add_argument("--center_b", type=float, default=-30.0, help="Center latitude (deg) in Galactic (b)")
    ap.add_argument("--patch_deg", type=float, default=30.0, help="Patch size (deg)")  # Zvětšeno pro lepší Annulus
    ap.add_argument("--pixsize_arcmin", type=float, default=5.0, help="Pixel size (arcmin)")
    ap.add_argument("--theta_c_deg", type=float, default=5.0, help="β-profile core radius (deg)")
    ap.add_argument("--theta_max_deg", type=float, default=15.0, help="β-profile truncation radius (deg)")
    ap.add_argument("--interp", choices=["nearest", "bilinear"], default="bilinear", help="Sampling interpolation")
    ap.add_argument("--save_prefix", type=str, default="coldspot_analysis", help="Output filename prefix")
    args = ap.parse_args()

    m, nside, ordering, coordsys = read_healpix_temperature(args.fits)
    print(f"Loaded HEALPix map: NSIDE={nside}, ORDERING={ordering}, COORDSYS={coordsys}")

    hp = build_healpix_sampler(nside, ordering, frame='galactic')

    # 1. Extrakce výřezu
    patch, X, Y = make_tangent_patch(hp, m, args.center_l, args.center_b,
                                     patch_deg=args.patch_deg,
                                     pixsize_arcmin=args.pixsize_arcmin,
                                     interp=args.interp)

    # 2. Vytvoření šablony
    templ = beta_profile_2d(X, Y, theta_c_deg=args.theta_c_deg, theta_max_deg=args.theta_max_deg)

    # 3. Matched Filter (ZNCC)
    response, corrmap = zncc_fft_match(patch, templ)
    print("\n" + "=" * 40)
    print(f"**Matched-filter (ZNCC) response at center: {response:.3f}**")
    print("=" * 40)

    # 4. NOVÁ ANALÝZA: Fisherova Kurtóza (Ne-Gaussovské testování)
    try:
        kurt_results = local_kurtosis_analysis(patch, templ)

        K_center = kurt_results.get('K_center', np.nan)
        K_annulus = kurt_results.get('K_annulus', np.nan)
        N_center = kurt_results.get('N_center', 0)

        print("\n=== Fisher Kurtosis Analysis (Ne-Gaussovita) ===")
        print(f"K_Center (Anomálie, N={N_center}): {K_center:.3f}")
        print(f"K_Annulus (Pozadí, N={kurt_results.get('N_annulus', 0)}): {K_annulus:.3f}")

        print("\n--- Interpretace ---")
        if np.isnan(K_center) or np.isnan(K_annulus):
            print("Chyba: Nedostatečná velikost vzorku pro spolehlivý výpočet kurtózy.")
        elif K_center > 0.1 and K_center > (K_annulus + 0.1):
            # Subjektivní prah pro 'signifikantně' kladnou kurtózu
            print("ZÁVĚR: CENTRÁLNÍ KURTÓZA JE VÝRAZNĚ KLADNÁ.")
            print("=> Podpora pro hypotézu Tachyonového Echa/Textury (Pole s těžkými ocasy, K > 0).")
        elif np.abs(K_center) < 0.1:
            print("ZÁVĚR: CENTRÁLNÍ KURTÓZA JE BLÍZKÁ NULE.")
            print("=> Podpora pro Supervoid/ISW efekt (Gaussovská fluktuace, K ≈ 0).")
        else:
            print("Kurtóza je mírná/nekonkluzivní.")

    except Exception as e:
        print(f"Kurtosis analysis failed: {e}")
    # ==================================

    # 5. Plotting (beze změny)

    fig1 = plt.figure(figsize=(6, 5))
    plt.imshow(patch, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel("Δl (deg)")
    plt.ylabel("Δb (deg)")
    plt.title("CMB Patch (μK)")
    plt.colorbar(label="μK")
    fig1.tight_layout()
    fig1.savefig(f"{args.save_prefix}_map.png", dpi=150)

    fig2 = plt.figure(figsize=(6, 5))
    plt.imshow(templ, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel("Δl (deg)")
    plt.ylabel("Δb (deg)")
    plt.title("β-profile Template (unit peak)")
    plt.colorbar()
    fig2.tight_layout()
    fig2.savefig(f"{args.save_prefix}_template.png", dpi=150)

    fig3 = plt.figure(figsize=(6, 5))
    plt.imshow(corrmap, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel("Δl (deg)")
    plt.ylabel("Δb (deg)")
    plt.title("ZNCC Correlation Map")
    plt.colorbar()
    fig3.tight_layout()
    fig3.savefig(f"{args.save_prefix}_corr.png", dpi=150)

    print(f"\nSaved plots to files starting with: {args.save_prefix}")


if __name__ == "__main__":
    main()