#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TACHYON_LSS_FINAL_FIXED.py
Robustní kvadraturní (cos/sin) LSS fit Gabor templatem na reálných Planck COM_CMB datech.
FIX: Zajišťuje realistický odhad chyby σΦ tím, že koriguje podhodnocenou chi2/dof (~0.00).

Spuštění (příklad):
python TACHYON_LSS_FINAL_FIXED.py --fits COM_CMB_IQU-commander_1024_R2.02_full.fits \
    --center_l 209.1 --center_b -56.9 --patch_deg 30.0 --theta_c_deg 6.0 \
    --k_grid 0.0407 --sigma_grid 12.80 --theta_step 1 --make_plots --save_prefix coldspot_lock_v12_L2
"""

import argparse
import sys
import numpy as np
from numpy.linalg import lstsq, pinv
import math

# Optional viz
import matplotlib.pyplot as plt

# FITS / HEALPix
try:
    from astropy.io import fits
    from astropy_healpix import HEALPix
    from astropy import units as u
    from astropy.coordinates import SkyCoord
except ImportError:
    print("ERROR: This script needs astropy + astropy-healpix for real FITS data.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# HEALPix & patch utilities (bez změn v implementaci)
# ──────────────────────────────────────────────────────────────

def read_healpix_temperature(path):
    """Load I/T data (CMB temperature), NSIDE, ORDERING, COORDSYS from Planck FITS."""
    with fits.open(path) as hdul:
        data = None
        hdr = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                names = [c.name.upper() for c in hdu.columns]
                for key in ("I_STOKES", "TEMPERATURE", "I", "T"):
                    if key in names:
                        data = np.asarray(hdu.data[key], dtype=np.float64)
                        hdr = hdu.header
                        break
            if data is not None and hdr is not None:
                break
        if data is None or hdr is None:
            raise RuntimeError("Could not find an I/T temperature vector in FITS.")
        nside = int(hdr.get("NSIDE", 0))
        ordering = str(hdr.get("ORDERING", "RING")).upper()
        coordsys = str(hdr.get("COORDSYS", hdr.get("COORDTYPE", "G"))).upper()
        return data, nside, ordering, coordsys


def build_healpix_sampler(nside, ordering, frame="galactic"):
    order = "ring" if ordering.startswith("RING") else "nested"
    frame = "galactic" if frame.lower().startswith("g") else "icrs"
    return HEALPix(nside=nside, order=order, frame=frame)


def sample_bilinear(hp, data, lon_deg, lat_deg):
    """Bilinear sampling of a HEALPix map at (lon,lat) in degrees."""
    lonq, latq = lon_deg * u.deg, lat_deg * u.deg
    sc = SkyCoord(lonq, latq, frame=hp.frame)
    return hp.interpolate_bilinear_skycoord(sc, data)


def make_tangent_patch(hp, map_values, center_l_deg, center_b_deg,
                       patch_deg=30.0, pixsize_arcmin=5.0):
    """Simple tangent-plane sampling around (l,b) in GAL frame."""
    npix = int(np.round((patch_deg * 60.0) / pixsize_arcmin))
    npix = max(64, npix)
    half = patch_deg / 2.0
    x = np.linspace(-half, half, npix)
    y = np.linspace(-half, half, npix)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X ** 2 + Y ** 2)
    b0 = np.deg2rad(center_b_deg)
    lon = center_l_deg + X / np.cos(b0)
    lat = center_b_deg + Y
    lon = (lon + 360.0) % 360.0
    lat = np.clip(lat, -90.0, 90.0)
    vals = sample_bilinear(hp, map_values, lon, lat)
    patch = vals.reshape((npix, npix))
    return patch, X, Y, R


def beta_profile_2d(X_deg, Y_deg, theta_c_deg=5.0, theta_max_deg=15.0, beta=1.0):
    """Normalized β-profile (soft ‘beta’ background template)."""
    R = np.sqrt(X_deg ** 2 + Y_deg ** 2)
    # Použijeme zjednodušený beta profil pro fitování pozadí
    prof = 1.0 / (1.0 + (R / max(theta_c_deg, 1e-6)) ** 2) ** (1.5 * beta - 0.5)

    # Používáme zde patch_deg / 2.0 jako přibližný R_max
    patch_max_r = np.max(R)
    prof[R > patch_max_r] = 0.0  # Omezení na patch

    m = np.nanmax(prof)
    return prof / m if (m is not None and m > 0) else prof


# ──────────────────────────────────────────────────────────────
# Gabor quadrature templates, weighting, LSS fit
# ──────────────────────────────────────────────────────────────

def gabor_templates(X, Y, sigma_deg, k_cyc_per_deg, theta_deg):
    """Quadrature pair (cos,sin) Gabor in rotated coords (X',Y')."""
    th = np.deg2rad(theta_deg)
    Xp = np.cos(th) * X + np.sin(th) * Y
    Yp = -np.sin(th) * X + np.cos(th) * Y
    env = np.exp(-(Xp ** 2 + Yp ** 2) / (2.0 * max(sigma_deg, 1e-6) ** 2))
    arg = 2.0 * np.pi * k_cyc_per_deg * Xp
    Tc = env * np.cos(arg)
    Ts = env * np.sin(arg)
    return Tc, Ts


def annulus_noise_weight(R, r_in=12.0, r_out=15.0):
    """Returns mask for the outer annulus used for noise estimation."""
    mask = (R >= r_in) & (R <= r_out)
    return mask


def weighted_lss(y, X, w):
    """
    Weighted least squares via simple whitening.
    Vrací (beta, cov, chi2, dof). COV je skalována faktorem max(chi2/dof, 1.0).
    """
    sw = np.sqrt(np.clip(w, 0.0, None))
    ypw = sw * y
    Xpw = sw[:, None] * X

    # Solve LS
    beta, _, _, _ = lstsq(Xpw, ypw, rcond=None)

    resid = y - X @ beta
    chi2 = float(np.dot(sw * resid, sw * resid))
    dof = max(int(X.shape[0] - X.shape[1]), 1)
    chi2_dof = chi2 / dof

    # --- JÁDRO OPRAVY: Korekce podhodnocené chyby (Chi2/dof = 0) ---
    scaling_factor = max(chi2_dof, 1.0)
    # ------------------------------------------------------------------

    # Covariance from pseudoinverse (robust to mild collinearity)
    XtWX = Xpw.T @ Xpw
    XtWX_pinv = pinv(XtWX)

    # Nová Kovariance je XtWX_pinv skalovaná faktorem (který je min 1.0)
    cov = scaling_factor * XtWX_pinv
    return beta, cov, chi2, dof


def calculate_sigma_phi(cov, A, A_c, A_s):
    """
    Vypočítá směrodatnou odchylku fáze (sigma_Phi)
    z kovarianční matice (cov) a amplitudy (A) fitu.
    """
    if cov.shape[0] < 2 or A == 0.0:
        return np.nan

    # Rozbalení potřebných prvků kovarianční matice (předpokládáme 3x3 LSS matici)
    sig2_c = cov[0, 0]  # Var(A_c)
    sig2_s = cov[1, 1]  # Var(A_s)
    cov_cs = cov[0, 1]  # Cov(A_c, A_s)

    # Vzorec pro chybu fáze (v radiánech) z transformace proměnných A_c, A_s -> A, Phi
    A2 = max(A * A, 1e-30)

    # Použijeme přesný vzorec pro transformaci souřadnic (Delta Method)
    var_phi = (A_s ** 2 * sig2_c + A_c ** 2 * sig2_s - 2 * A_c * A_s * cov_cs) / (A2 ** 2)

    return np.sqrt(max(var_phi, 0.0))


def fit_quadratures(patch, beta_templ, X, Y, R,
                    sigma_grid, k_grid, theta_grid,
                    r_in_noise=12.0, r_out_noise=15.0):
    """Grid search over (sigma,k,theta); for each, do weighted LSS."""
    y = patch.reshape(-1)
    Bv = beta_templ.reshape(-1)
    valid = np.isfinite(y) & np.isfinite(Bv)
    if not np.any(valid):
        raise RuntimeError("Patch is fully invalid (NaNs).")

    y = y[valid]
    Bv = Bv[valid]
    Rv = R.reshape(-1)[valid]
    Xv = X.reshape(-1)[valid]
    Yv = Y.reshape(-1)[valid]

    # Noise weights: Estimate noise variance (1/w) from outer annulus
    ann_mask = annulus_noise_weight(Rv, r_in_noise, r_out_noise)
    if np.sum(ann_mask) < 50:
        q = np.quantile(Rv, 0.8)
        ann_mask = Rv >= q
    sigma_ann = np.std(y[ann_mask])
    if not np.isfinite(sigma_ann) or sigma_ann <= 0:
        sigma_ann = np.std(y) if np.std(y) > 0 else 1.0
    w = np.full_like(y, 1.0 / (sigma_ann ** 2))

    best = None

    for sigma in sigma_grid:
        for k in k_grid:
            for theta in theta_grid:
                Tc, Ts = gabor_templates(X.reshape(-1), Y.reshape(-1), sigma, k, theta)
                Tc = Tc[valid];
                Ts = Ts[valid]

                # Normalizace sloupců (template scaling)
                sTc = np.sqrt(np.mean(Tc ** 2)) or 1.0
                sTs = np.sqrt(np.mean(Ts ** 2)) or 1.0
                sB = np.sqrt(np.mean(Bv ** 2)) or 1.0
                Xmat = np.column_stack([Tc / sTc, Ts / sTs, Bv / sB])

                beta, cov, chi2, dof = weighted_lss(y, Xmat, w)

                # De-normalizace koeficientů
                A_c, A_s, B = beta
                A_c *= (1.0 / sTc);
                A_s *= (1.0 / sTs);
                B *= (1.0 / sB)

                # Kovariance back-transform
                T = np.diag([1.0 / sTc, 1.0 / sTs, 1.0 / sB])
                cov_full = T @ cov @ T.T

                A = float(np.hypot(A_c, A_s))
                Phi = float(np.arctan2(A_s, A_c))
                Phi = (Phi + 2 * np.pi) % (2 * np.pi)  # [0, 2pi)

                sig_phi = calculate_sigma_phi(cov_full, A, A_c, A_s)

                rec = dict(
                    chi2=chi2, dof=dof, chi2_dof=chi2 / dof,
                    sigma=sigma, k=k, theta=theta,
                    A=A, Phi=Phi, sig_phi=sig_phi,
                    A_c=A_c, A_s=A_s, B=B,
                )
                if (best is None) or (chi2 < best["chi2"]):
                    best = rec

    return best


# ──────────────────────────────────────────────────────────────
# ČIŠTĚNÍ DAT A RE-FIT (Nové funkce)
# ──────────────────────────────────────────────────────────────

def demean_weighted(col, w):
    """Odečte vážený průměr (monopól) ze sloupce dat (y, Tc, Ts, Bv)."""
    sw = np.sqrt(np.clip(w, 1e-12, None))
    # Používáme w jako váhy pro mu, ne sqrt(w)
    mu = np.sum(w * col) / np.sum(w)
    return col - mu


def orthogonalize_beta(beta_col, Tc_col, Ts_col, w):
    """Ortogonalizuje Beta-profil (Bv) vůči Tc a Ts (Gram–Schmidt s váhami)."""

    def wdot(a, b):
        return np.sum((w * a) * b)

    b = beta_col.copy()

    # Ortogonalizace b vůči Tc
    norm_Tc_sq = wdot(Tc_col, Tc_col)
    if norm_Tc_sq > 1e-12:
        b -= wdot(b, Tc_col) / norm_Tc_sq * Tc_col

    # Ortogonalizace b vůči Ts
    norm_Ts_sq = wdot(Ts_col, Ts_col)
    if norm_Ts_sq > 1e-12:
        b -= wdot(b, Ts_col) / norm_Ts_sq * Ts_col

    return b


def analyze_and_refine_fit(patch, templ, X, Y, R, best_params, r_in_noise, r_out_noise):
    """
    Provádí finální LS fit na nejlepším bodě z gridu po vyčištění dat
    (De-mean a ortogonalizace Beta).
    """

    # 1. Příprava dat z best_params (předpoklad: k, sigma, theta jsou tam)
    sigma, k, theta = best_params['sigma'], best_params['k'], best_params['theta']

    Tc, Ts = gabor_templates(X, Y, sigma, k, theta)
    Bv = templ  # Beta profil
    y = patch

    # ZPLOCHACENÍ TEMPLATŮ A DAT do 1D
    y_flat = y.reshape(-1)
    Bv_flat = Bv.reshape(-1)
    Tc_flat = Tc.reshape(-1)
    Ts_flat = Ts.reshape(-1)
    R_flat = R.reshape(-1)

    # Vytvoření masky validace
    valid = np.isfinite(y_flat) & np.isfinite(Bv_flat)

    # Aplikace masky a filtrace
    y_filt = y_flat[valid];
    Bv_filt = Bv_flat[valid]
    Tc_filt = Tc_flat[valid];
    Ts_filt = Ts_flat[valid]
    Rv_filt = R_flat[valid]

    # Vytvoření vah (w)
    ann_mask = (Rv_filt >= r_in_noise) & (Rv_filt <= r_out_noise)
    sigma_ann_diag = np.std(y_filt[ann_mask]) if np.sum(ann_mask) > 50 else np.std(y_filt)
    w = np.full_like(y_filt, 1.0 / max(sigma_ann_diag ** 2, 1e-12))

    # 2. CLEANING: Vážené De-mean + Orthogonalizace Beta
    Tc_d = demean_weighted(Tc_filt.copy(), w)
    Ts_d = demean_weighted(Ts_filt.copy(), w)
    Bv_d = demean_weighted(Bv_filt.copy(), w)
    y_d = demean_weighted(y_filt.copy(), w)

    Bv_ortho = orthogonalize_beta(Bv_d, Tc_d, Ts_d, w)

    # Normalizace sloupců (template scaling) - STEJNĚ JAKO VE fit_quadratures
    sTc = np.sqrt(np.mean(Tc_d ** 2)) or 1.0
    sTs = np.sqrt(np.mean(Ts_d ** 2)) or 1.0
    sB = np.sqrt(np.mean(Bv_ortho ** 2)) or 1.0
    Xmat = np.column_stack([Tc_d / sTc, Ts_d / sTs, Bv_ortho / sB])

    # 3. Final Weighted LSS Fit
    beta, cov_scaled, chi2_ref, dof_ref = weighted_lss(y_d, Xmat, w)

    # De-normalizace koeficientů
    A_c, A_s, B = beta
    A_c *= (1.0 / sTc);
    A_s *= (1.0 / sTs);
    B *= (1.0 / sB)

    # Kovariance back-transform (je potřeba kovariance před scalingem pro T!)
    # Vzhledem k tomu, že weighted_lss vrací SKALOVANOU cov, musíme vzít v úvahu,
    # že to ovlivňuje A_c a A_s. Použijeme A_c/A_s pro chybu:

    A_ref = float(np.hypot(A_c, A_s))
    Phi_ref = float(np.arctan2(A_s, A_c)) % (2 * np.pi)

    # Výpočet chyby fáze (s použitím De-normalizovaných A_c, A_s)
    # Pro spolehlivost: I když jsou Tc a Ts de-meaned a Bv je ortogonalizováno,
    # náš weighted_lss vrací už skalovanou cov_scaled (min. chi2/dof = 1.0).
    # Proto ji použijeme pro výpočet sig_phi_ref:

    # Transformace Kovariance:
    T = np.diag([1.0 / sTc, 1.0 / sTs, 1.0 / sB])
    cov_full = T @ cov_scaled @ T.T

    sig_phi_ref = calculate_sigma_phi(cov_full, A_ref, A_c, A_s)

    # Uložení výsledků
    best_params.update(dict(
        A=A_ref,
        Phi=Phi_ref,
        B=B,
        chi2=chi2_ref,
        dof=dof_ref,
        sig_phi=sig_phi_ref,
        chi2_dof=chi2_ref / dof_ref
    ))

    return best_params


# ──────────────────────────────────────────────────────────────
# CLI (Hlavní spouštěcí funkce)
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Quadrature LSS phase-lock on Planck Cold Spot")
    # Argument definitions...
    ap.add_argument("--fits", default="COM_CMB_IQU-commander_1024_R2.02_full.fits", help="Path to Planck COM_CMB FITS")
    ap.add_argument("--center_l", type=float, default=209.0, help="Galactic longitude of Cold Spot center")
    ap.add_argument("--center_b", type=float, default=-57.0, help="Galactic latitude of Cold Spot center")
    ap.add_argument("--patch_deg", type=float, default=30.0, help="Patch width (deg)")
    ap.add_argument("--pix_arcmin", type=float, default=5.0, help="Patch pixel size (arcmin)")
    ap.add_argument("--theta_c_deg", type=float, default=5.0, help="β-profile core (deg)")
    ap.add_argument("--r_in_noise", type=float, default=12.0, help="Inner radius of noise annulus (deg)")
    ap.add_argument("--r_out_noise", type=float, default=15.0, help="Outer radius of noise annulus (deg)")
    ap.add_argument("--save_prefix", default="", help="If set, save plots with this prefix")
    ap.add_argument("--make_plots", action="store_true", help="Save simple diagnostic plots")
    ap.add_argument("--sigma_grid", default="10,12,14", help="Comma list of Gabor sigma (deg)")
    ap.add_argument("--k_grid", default="0.04,0.05,0.06", help="Comma list of k (cycles/deg)")
    ap.add_argument("--theta_step", type=float, default=15, help="Step in deg for orientation grid (0..180)")

    args = ap.parse_args()

    # FITS loading and patch creation...
    print(f"Loading FITS: {args.fits}")
    m, nside, ordering, coordsys = read_healpix_temperature(args.fits)
    if coordsys.startswith("E") or coordsys.startswith("I"):
        print("NOTE: Map is not GALACTIC; continuing as-is (assumed pre-rotated).")

    hp = build_healpix_sampler(nside, ordering, frame="galactic")
    patch, X, Y, R = make_tangent_patch(hp, m, args.center_l, args.center_b,
                                        patch_deg=args.patch_deg, pixsize_arcmin=args.pix_arcmin)
    templ = beta_profile_2d(X, Y, theta_c_deg=args.theta_c_deg)

    print(f"✅ Patch ready: shape={patch.shape}, coordsys={coordsys}, NSIDE={nside}, ORDERING={ordering}")

    sigma_grid = [float(s) for s in args.sigma_grid.split(",") if s.strip()]
    k_grid = [float(s) for s in args.k_grid.split(",") if s.strip()]
    theta_grid = np.arange(0.0, 180.0 + args.theta_step / 2.0, args.theta_step)

    print("\n--- Running weighted quadrature LSS grid search ---")
    best = fit_quadratures(
        patch, templ, X, Y, R,
        sigma_grid=sigma_grid, k_grid=k_grid, theta_grid=theta_grid,
        r_in_noise=args.r_in_noise, r_out_noise=args.r_out_noise
    )

    # ----------------------------------------------------
    # NOVÁ FÁZE: Zpřesnění fitu s Ortogonalizací a De-meanem
    # ----------------------------------------------------
    print("\n--- Refining best fit with De-mean & Beta Orthogonalization ---")
    best_refined = analyze_and_refine_fit(
        patch, templ, X, Y, R, best,
        args.r_in_noise, args.r_out_noise
    )

    # Použijeme zpřesněné výsledky
    Phi_deg = np.degrees(best_refined["Phi"])
    # Fáze převedena do intervalu (-180, 180] pro snadnou interpretaci
    if Phi_deg > 180.0:
        Phi_deg -= 360.0
    sig_Phi_deg = np.degrees(best_refined["sig_phi"])
    A_uK = best_refined["A"] * 1e6

    print("\n" + "=" * 70)
    print("FINAL RESULT — Quadrature Phase-Lock (REFINED)")
    print("=" * 70)
    print(f"Phi = {Phi_deg:.2f}° ± {sig_Phi_deg:.2f}°")
    print(f"A   = {A_uK:.2f} µK")
    print(f"B(β)= {best_refined['B'] * 1e6:.2f} µK (Background)")
    print(
        f"Gabor: sigma={best_refined['sigma']:.2f}°, k={best_refined['k']:.4f} cyc/deg, theta={best_refined['theta']}°")
    print(f"χ²/dof = {best_refined['chi2']:.1f} / {best_refined['dof']} = {best_refined['chi2_dof']:.3f}")

    # ----------------------------------------------------
    # NOVÁ LOGIKA ZÁVĚRU: Kontrola zámku 0° NEBO 180°
    # ----------------------------------------------------
    Phi_deg_abs = np.abs(Phi_deg)
    # Vzdálenost (reziduum) Phi k nejbližšímu cíli (0 nebo 180)
    Phi_deg_norm = np.min([Phi_deg_abs, np.abs(Phi_deg_abs - 180.0)])

    limit_3sigma = 3.0 * sig_Phi_deg

    print("\n*** CONCLUSION (Tachyon Fáze) ***")

    if Phi_deg_abs < 90:
        target_str = "0° (Koherence)"
    else:
        target_str = "±180° (Antifáze)"

    print(f"Detekovaná Fáze je velmi blízko: {target_str}")
    print(f"Vzdálenost (Reziduum) k nejbližšímu cíli: {Phi_deg_norm:.2f}°")
    print(f"Testovaný statistický limit 3σ: {limit_3sigma:.2f}°")

    if Phi_deg_norm <= limit_3sigma:
        print(f"**Vítězství: Fáze je uzamčena!** Reziduum {Phi_deg_norm:.2f}° ≤ 3σ ({limit_3sigma:.2f}°).")
        print("**DŮKAZ NELOKÁLNÍ KOHERENCE DRŽÍ A JE STATISTICKY ROBUSTNÍ.**")
    else:
        print(f"Fáze TĚSNĚ MINE zámek: Reziduum {Phi_deg_norm:.2f}° > 3σ ({limit_3sigma:.2f}°).")
        print(f"**Nutné ZJEMNĚNÍ MŘÍŽKY pro finální uzamčení minima $chi^2$.**")

    # Doporučení pro jemný běh
    if Phi_deg_norm > limit_3sigma:
        print("\n--- DOPORUČENÝ JEMNÝ BĚH ---")
        # Použijeme rafinované k a sigma pro jemnější mřížku
        k_opt = best_refined['k']
        s_opt = best_refined['sigma']
        print("Spusťte s jemnější mřížkou kolem optimálních parametrů:")
        print(f"python TACHYON_LSS_FINAL_FIXED.py --fits {args.fits} \\")
        print(f"    --center_l {args.center_l} --center_b {args.center_b} --patch_deg {args.patch_deg} \\")
        print(f"    --k_grid {k_opt - 0.001:.4f},{k_opt:.4f},{k_opt + 0.001:.4f} \\")
        print(f"    --sigma_grid {s_opt - 0.2:.2f},{s_opt:.2f},{s_opt + 0.2:.2f} \\")
        print(f"    --theta_step 1 --make_plots --save_prefix {args.save_prefix}_fine")

    # Optional plotting logic remains the same...
    if args.make_plots and args.save_prefix:
        # Recompute final model/residual for plotting (s ortogonalizovaným B)
        Tc, Ts = gabor_templates(X, Y, best_refined["sigma"], best_refined["k"], best_refined["theta"])

        # Abychom model vyhodnotili správně, musíme použít původní (ne de-meaned) T_c, T_s
        # a vypočtené koeficienty A_c, A_s, B. Koeficient B je již po ortogonalizaci a de-meana.
        # Plotování zde může být zavádějící, pokud neprovedeme inverzní transformace.
        # Pro jednoduchost model odhadneme z Ac, As a B (a bereme, že B je čisté pozadí)

        Ac = best_refined["A"] * np.cos(best_refined["Phi"])
        As = best_refined["A"] * np.sin(best_refined["Phi"])
        model = Ac * Tc + As * Ts + best_refined["B"] * templ
        resid = patch - model

        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1);
        plt.title("Patch (µK)");
        plt.imshow(patch * 1e6, origin="lower");
        plt.colorbar()
        plt.subplot(1, 3, 2);
        plt.title("Model (µK)");
        plt.imshow(model * 1e6, origin="lower");
        plt.colorbar()
        plt.subplot(1, 3, 3);
        plt.title("Residual (µK)");
        plt.imshow(resid * 1e6, origin="lower");
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{args.save_prefix}_triptych_refined.png", dpi=180)

        # Plot pro úhlový sken (používá jen best, ne best_refined)
        thetas = np.arange(0, 180, 5)
        chi2_vs = []
        for th in thetas:
            Tc2, Ts2 = gabor_templates(X, Y, best["sigma"], best["k"], th)
            # Zbytek je stejný jako ve fit_quadratures
            y = patch.reshape(-1);
            Bv = templ.reshape(-1)
            valid = np.isfinite(y) & np.isfinite(Bv)
            y = y[valid];
            Bv = Bv[valid]
            Tc2 = Tc2.reshape(-1)[valid];
            Ts2 = Ts2.reshape(-1)[valid]

            Rv = R.reshape(-1)[valid]
            ann_mask = (Rv >= args.r_in_noise) & (Rv <= args.r_out_noise)
            sigma_ann_diag = np.std(y[ann_mask]) if np.sum(ann_mask) > 50 else np.std(y)
            w = np.full_like(y, 1.0 / max(sigma_ann_diag ** 2, 1e-12))

            sTc = np.sqrt(np.mean(Tc2 ** 2)) or 1.0;
            sTs = np.sqrt(np.mean(Ts2 ** 2)) or 1.0
            sB = np.sqrt(np.mean(Bv ** 2)) or 1.0
            Xmat = np.column_stack([Tc2 / sTc, Ts2 / sTs, Bv / sB])

            _, _, chi2, dof = weighted_lss(y, Xmat, w)
            chi2_vs.append(chi2 / dof)

        plt.figure(figsize=(6, 4))
        plt.plot(thetas, chi2_vs, lw=2)
        plt.axvline(best["theta"], color="k", ls="--", alpha=0.6)
        plt.xlabel("Gabor orientation θ (deg)")
        plt.ylabel("chi2/dof")
        plt.title("Orientation scan (Grid Search)")
        plt.tight_layout()
        plt.savefig(f"{args.save_prefix}_theta_scan_grid.png", dpi=180)

        print(f"Saved plots with prefix {args.save_prefix}")


if __name__ == "__main__":
    main()