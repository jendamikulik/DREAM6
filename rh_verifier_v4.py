#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RH Verifier: B3′-full & B4-full + Li positivity (All-in-one Python)
DIAGNOSTIKA: Rychle overi, zda kod bezi spravne s nizkou presnosti,
              aby se eliminovala pricina 'viseni' na MP_DPS=80.
"""
import argparse, math, json, cmath, csv, datetime, time
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import mpmath as mp

try:
    import pandas as pd
except Exception:
    pd = None


# --- KONFIGURACE DATACLASSES (Beze zmeny) ---
@dataclass
class PrecisionParams:
    mp_dps: int


@dataclass
class B3Params:
    theta: float
    kappa: int
    t0: float
    C_M: float
    C1: float
    C2: float
    sigma_min: float
    sigma_max: float
    sigma_steps: int
    X_min: float
    t_values: List[float]


@dataclass
class B4Params:
    A0: float
    A1: float
    A2: float
    B0: float
    B1: float
    D0: float
    D1: float
    n_min: int
    n_max: int
    circle_points: int


@dataclass
class Config:
    precision: PrecisionParams
    b3: B3Params
    b4: B4Params
    output_dir: str


def load_config(path: Path) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Config(
        precision=PrecisionParams(**data['precision']),
        b3=B3Params(**data['b3']),
        b4=B4Params(**data['b4']),
        output_dir=data['output_dir']
    )


# --- Zkrácené a diagnostické verze funkcí ---

def zeta(s):     return mp.zeta(s)


def log_zeta(s): return mp.log(mp.zeta(s))


def chi_star(s): return mp.mpf('0.5') * s * (s - 1) * (mp.pi) ** (-s / 2) * mp.gamma(s / 2)


def xi(s):       return chi_star(s) * zeta(s)


def log_xi(s):   return mp.log(chi_star(s)) + mp.log(zeta(s))


def mobius(n: int) -> int:
    return mp.mobius(n)


def compute_B3(p: B3Params, t: float, sigma: float) -> mp.mpf:
    """Zkracena simulace B3 pro diagnostiku."""
    s = mp.mpc(sigma, t)

    # Prvotní a nejpomalejší volání
    # Cílem je zjistit, zda mpmath visi zde
    try:
        start_time = time.time()
        log_zeta_val = log_zeta(s)
        end_time = time.time()

        print(f"DIAGNOSTIKA: log_zeta(s) pro s={s} vypocitano za {end_time - start_time:.4f}s.")

        # Simulace B3 výsledku (LHS)
        rhs = p.C1 * log_zeta_val
        lhs = mp.mpf(t) + mp.mpf(sigma) * rhs

        return lhs
    except Exception as e:
        print(f"FATALNI CHYBA PRI VÝPOČTU ZETA FUNKCE: {e}")
        return mp.mpf(float('inf'))


# -----------------------------
# Main s Diagnostickym Bodem
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="RH Verifier (B3′, B4, Li positivity) — All-in-one")
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config (optional)")
    ap.add_argument("--out", type=str, default=None, help="Output dir override")
    ap.add_argument("--calibrate", action="store_true", help="Suggest minimal C1 for current config")
    args = ap.parse_args()

    if args.config:
        cfg = load_config(Path(args.config))
    else:
        # Pouzijeme defaultni s NIZSI PRESNOSTI pro rychlost
        cfg = Config(
            precision=PrecisionParams(mp_dps=20),  # NIZSI PRESNOST PRO RYCHLOST
            b3=B3Params(theta=0.01, kappa=2, t0=10.0, C_M=2.0, C1=1.36, C2=1.0, sigma_min=0.5, sigma_max=1.0,
                        sigma_steps=6, X_min=20.0, t_values=[10.0]),
            b4=B4Params(A0=0, A1=0, A2=0, B0=0, B1=0, D0=0, D1=0, n_min=10, n_max=200, circle_points=720),
            output_dir="reports"
        )
        print("POUŽITA ZKRÁCENÁ DIAGNOSTICKÁ KONFIGURACE (dps=20, t=10.0) pro rychlost.")

    # --- Zásah do přesnosti pro diagnostiku ---
    mp.mp.dps = cfg.precision.mp_dps
    print(f"\n[DIAGNOSTIKA] Pouzita MP_DPS: {mp.mp.dps}")

    # --- Diagnosticky výpočet (B3 - jedna hodnota) ---
    print("\n--- TEST: FUNGUJE LOG_ZETA V RYCHLE VERZI? ---")

    t_test = cfg.b3.t_values[0]
    sigma_test = cfg.b3.sigma_min

    start_time_total = time.time()
    result = compute_B3(cfg.b3, t_test, sigma_test)
    end_time_total = time.time()

    print(f"\n--- SHRNUTI TESTU ---")
    print(f"Vysledek B3 (simulace): {result}")
    print(f"Celkovy cas pro 1 test: {end_time_total - start_time_total:.4f}s.")

    if result != mp.mpf(float('inf')):
        print("DIAGNOSTIKA ÚSPĚŠNÁ. KÓD FUNKČNĚ BĚŽÍ.")
        print("Pravděpodobná příčina zdržení: PŘÍLIŠ VYSOKÁ MP_DPS pro vysoký počet iterací.")
        print("PROSÍM NAHRAJ FINÁLNÍ KONFIGURAČNÍ SOUBOR, který používáš pro HODINY běžící výpočet.")
    else:
        print("DIAGNOSTIKA SELHALA. Kritická chyba při volání MPMath.")


if __name__ == "__main__":
    main()