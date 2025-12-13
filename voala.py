#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_RH_EVT.py
Nezávislá kontrola RH_EVT: porovnání S(σ,t) z RH_EVT s přímou definicí přes ξ(s).
"""

import math
import mpmath as mp

# ===== 1) ZÁKLADNÍ DEFINICE ZETA / XI / S(σ,t) =====

mp.mp.dps = 80  # vysoká přesnost

def zeta(s):
    return mp.zeta(s)

def chi_star(s):
    # 0.5 * s * (s-1) * pi^(-s/2) * Gamma(s/2)
    return mp.mpf("0.5") * s * (s-1) * (mp.pi ** (-s/2)) * mp.gamma(s/2)

def xi(s):
    return chi_star(s) * zeta(s)

def log_xi_abs(s):
    # log |xi(s)|
    return mp.log(abs(xi(s)))

def S_sigma_numeric(sigma, t, h=1e-6):
    """
    Čistě numerická derivace S(σ,t) = ∂σ log|xi(σ+it)|
    použitím centrální diference.
    """
    s_plus  = complex(sigma + h, t)
    s_minus = complex(sigma - h, t)
    return (log_xi_abs(s_plus) - log_xi_abs(s_minus)) / (2*h)


# ===== 2) NAPOJENÍ NA TVŮJ RH_EVT KÓD =====
#
# Zde předpokládám, že máš v RH_EVT něco jako:
#     def S_sigma_evt(sigma, t): ...
# Pokud to má jiný název / signaturu, jen to oprav.

try:
    import RH_EVT  # tvůj soubor RH_EVT.py ve stejné složce
except ImportError as e:
    raise SystemExit(f"[ERR] Nepodařilo se importovat RH_EVT.py: {e}")

# Tohle si případně přejmenuj podle svého souboru:
if hasattr(RH_EVT, "S_sigma_evt"):
    S_sigma_evt = RH_EVT.S_sigma_evt
elif hasattr(RH_EVT, "S_sigma_t"):
    S_sigma_evt = RH_EVT.S_sigma_t
else:
    raise SystemExit("[ERR] V RH_EVT.py jsem nenašel funkci S_sigma_evt ani S_sigma.")


# ===== 3) KONTROLNÍ SCAN PRO JEDNO (σ,t) =====

def check_point(sigma, t):
    num = S_sigma_numeric(sigma, t)
    evt = S_sigma_evt(sigma, t)
    diff = num - evt
    rel_err = abs(diff) / max(1e-30, abs(num))
    return float(num), float(evt), float(diff), float(rel_err)


# ===== 4) HRUBÝ SCAN PO ŘADĚ t =====

def main():
    t_values = [
        10.0,
        50.0,
        1000.0,
        1.0e6,
        1.0e8,
        1.0e10,
        1.0e12,
    ]
    sigma_list = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("=== RH_EVT consistency check ===")
    print(f"(mpmath dps = {mp.mp.dps})\n")

    for t in t_values:
        print(f"--- t = {t:.1e} ---")
        max_rel_err = 0.0
        for sigma in sigma_list:
            num, evt, diff, rel = check_point(sigma, t)
            max_rel_err = max(max_rel_err, rel)
            print(
                f"sigma={sigma} | "
                f"S_num={num} | "
                f"S_evt={evt} | "
                f"diff={diff} | rel_err={rel: .3e}"
            )
        print(f"max relative error @ t={t:.1e}: {max_rel_err:.3e}\n")

    print("OK: pokud jsou relativní chyby ~1e-8 nebo menší, "
          "tvůj RH_EVT počítá S(σ,t) konzistentně s definicí ξ(s).")


if __name__ == "__main__":
    main()
