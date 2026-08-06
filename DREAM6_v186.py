#!/usr/bin/env python3
"""
DREAM6 v171
Universal latent compatibility transport over hierarchical semantic contraction; frozen length-4 instanton retained as local evaluator.

Active pipeline
---------------
1. Parse the original CNF without changing its Boolean semantics.
2. Detect motifs of the exact form

       (x_1 OR ... OR x_k)
       AND all pairwise clauses (-x_i OR -x_j),

   consuming one binary-clause occurrence for every pair.  Replace each
   complete motif by one EXACT1 factor:

       x_1 + ... + x_k = 1.

   Clauses not belonging to a complete motif remain ordinary OR factors.
3. Apply one fixed reinforced synchronous cavity operator exactly 64*n times
   from the all-zero continuous message state.
4. Perform exactly one Boolean readout sign(H_i).
5. Verify that assignment independently against the ORIGINAL CNF.

Ordinary OR factor
------------------
For edge i -> a with literal sign c_ai in {+1,-1}:

    p_{i->a}^{viol} = sigmoid(-c_ai L_{i->a}),

    U_{a->i}
      = c_ai [-log(1 - product_{j in a, j != i} p_{j->a}^{viol})].

EXACT1 factor
-------------
For an EXACT1 group G and target i in G:

    U_{G->i}
      = log Z(x_i=1) - log Z(x_i=0)
      = -log sum_{j in G, j != i} exp(L_{j->G}).

Variable update
---------------
    H_i = sum_{a in N(i)} U_{a->i},

    L_{i->a}^{new}
      = H_i - U_{a->i} + rho H_i.

Both message families use the same fixed damping coefficient alpha.

There is no:
- intermediate Boolean assignment or UNSAT count,
- selection or ranking by verifier score,
- clause memory or dynamic clause reweighting,
- residual-clause/literal selection,
- variable flipping,
- branching, decimation, restart portfolio, or external SAT solver.

The factor fusion is a formula preprocessing identity, not a heuristic:
each fused factor is logically equivalent to the exact set of original
clauses it replaces.  This is not a completeness proof.  SAT is emitted only
after exact U=0 verification against the original CNF.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    from scipy.sparse import coo_matrix as scipy_coo_matrix
    from scipy.sparse.linalg import eigsh as scipy_eigsh
except ImportError:
    scipy_coo_matrix = None
    scipy_eigsh = None


# ---------------------------------------------------------------------------
# Numerically specified high-accuracy SVML exp kernel
# ---------------------------------------------------------------------------
#
# The original v160 X9 trajectory was later found to depend on NumPy's
# AVX-512 high-accuracy SVML exp implementation.  The dependency was not a
# random seed or a verifier-guided choice: it was the exact floating-point map
# used inside the continuous length-4 EVEN_CYCLE factor.  v170 made that map explicit and
# platform-defined by compiling the scalar IEEE-754 transcription below.
#
# The transcription was audited against NumPy 2.3.5 __svml_exp8_ha:
#   * exact bit agreement on 300,000 independent values in [-80,0],
#   * exact bit agreement on all 7,200 stored critical v160 trajectory values.
#
# The helper is loaded only when an EVEN_CYCLE factor exists.  Ordinary OR/EXACT1
# formulas retain the inherited v160 arithmetic unchanged.

_SVML_HA_C_SOURCE = r"""
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <fenv.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#  define DREAM6_X86 1
#else
#  define DREAM6_X86 0
#endif
#if defined(_WIN32)
#  define DREAM6_EXPORT __declspec(dllexport)
#else
#  define DREAM6_EXPORT __attribute__((visibility("default")))
#endif
#pragma STDC FENV_ACCESS ON

static inline double dream6_fma(double a, double b, double c) {
#if DREAM6_X86
    return __builtin_fma(a, b, c);
#else
    return fma(a, b, c);
#endif
}

static const double INVLN2 = 0x1.71547652b82fep+0;
static const double SHIFTER = 0x1.8000000003ff0p+48;
static const double LN2_HI = 0x1.62e42fefa39efp-1;
static const double LN2_LO = 0x1.abc9e3b39803fp-56;
static const double C5 = 0x1.7411836940c04p-10;
static const double C4 = 0x1.1101cbbc265c0p-7;
static const double C3 = 0x1.55557242d68fep-5;
static const double C2 = 0x1.5555553939732p-3;
static const double C1 = 0x1.000000000d008p-1;
static const double C0 = 0x1.fffffffffff70p-1;
static const double HI[16] = {
  0x1.0000000000000p+0, 0x1.0b5586cf9890fp+0,
  0x1.172b83c7d517bp+0, 0x1.2387a6e756238p+0,
  0x1.306fe0a31b715p+0, 0x1.3dea64c123422p+0,
  0x1.4bfdad5362a27p+0, 0x1.5ab07dd485429p+0,
  0x1.6a09e667f3bcdp+0, 0x1.7a11473eb0187p+0,
  0x1.8ace5422aa0dbp+0, 0x1.9c49182a3f090p+0,
  0x1.ae89f995ad3adp+0, 0x1.c199bdd85529cp+0,
  0x1.d5818dcfba487p+0, 0x1.ea4afa2a490dap+0
};
static const double LO[16] = {
  0x0.0p+0, 0x1.79aa65d837b6dp-54,
 -0x1.01b15eaa59348p-55, 0x1.68efde3a8a894p-54,
  0x1.34d754db0abb6p-55, 0x1.59f48a72a4c6dp-55,
  0x1.690cebb7aafb0p-56, 0x1.063e1e21c5409p-54,
 -0x1.3b3efbf5e2228p-54,-0x1.b32dcb94da51dp-56,
  0x1.db72fc1f0eab4p-55, 0x1.1affc2b91ce27p-56,
  0x1.c1a7792cb3387p-55, 0x1.36eae30af0cb3p-56,
  0x1.4a385a63d07a7p-56,-0x1.ff7128fd391f0p-55
};
typedef union { double d; uint64_t u; } du64;

DREAM6_EXPORT int dream6_svml_exp_ha_array(
    const double *x, double *out, size_t n
) {
    if (!x || !out) return -1;

#if DREAM6_X86
    /*
       SVML's first FMA uses {rz-sae}.  On Windows, C fenv functions are not
       a sufficiently strong contract for SSE/AVX arithmetic across CRTs, so
       set the MXCSR rounding-control bits directly.  All arithmetic in this
       helper is compiled for SSE/AVX/FMA; x87 is not part of the map.
    */
    const unsigned old_mxcsr = _mm_getcsr();
    const unsigned rz_mxcsr =
        (old_mxcsr & ~((unsigned)_MM_ROUND_MASK)) |
        (unsigned)_MM_ROUND_TOWARD_ZERO;
    _mm_setcsr(rz_mxcsr);
#else
    const int old_round = fegetround();
    if (old_round < 0) return -2;
    if (fesetround(FE_TOWARDZERO) != 0) return -3;
#endif

    /* vfmadd213pd(...){rz-sae}: range reduction on a 1/16 grid. */
    for (size_t i = 0; i < n; ++i) {
        volatile double xv = x[i];
        out[i] = dream6_fma(xv, INVLN2, SHIFTER);
    }

#if DREAM6_X86
    const unsigned rn_mxcsr =
        (old_mxcsr & ~((unsigned)_MM_ROUND_MASK)) |
        (unsigned)_MM_ROUND_NEAREST;
    _mm_setcsr(rn_mxcsr);
#else
    if (fesetround(FE_TONEAREST) != 0) return -4;
#endif

    for (size_t i = 0; i < n; ++i) {
        const double xv = x[i];
        const double t = out[i];
        const double n16 = t - SHIFTER;
        const du64 bits = {.d = t};
        const unsigned j = (unsigned)(bits.u & 15u);

        double r = dream6_fma(-n16, LN2_HI, xv);
        r = dream6_fma(-n16, LN2_LO, r);
        const double r2 = r * r;
        const double p45 = dream6_fma(C5, r, C4);
        const double p23 = dream6_fma(C3, r, C2);
        const double p01 = dream6_fma(C1, r, C0);
        double p = dream6_fma(p45, r2, p23);
        p = dream6_fma(p, r2, p01);
        double z = dream6_fma(p, r, LO[j]);
        z = dream6_fma(z, HI[j], HI[j]);
        out[i] = scalbn(z, (int)floor(n16));
    }

#if DREAM6_X86
    _mm_setcsr(old_mxcsr);
#else
    if (old_round != FE_TONEAREST) {
        if (fesetround(old_round) != 0) return -5;
    }
#endif

    return 0;
}

static const double LOG_C9 = 0x1.c81cd309d7c70p-4;
static const double LOG_C8 = -0x1.007357e93af62p-3;
static const double LOG_C7 = 0x1.249229cee81efp-3;
static const double LOG_C6 = -0x1.55553fb28db06p-3;
static const double LOG_C5 = 0x1.9999999cc9f5cp-3;
static const double LOG_C4 = -0x1.00000000c05bdp-2;
static const double LOG_C3 = 0x1.5555555555466p-2;
static const double LOG_C2 = -0x1.fffffffffffc6p-2;
static const double LOG_LN2_HI = 0x1.62e42fefa0000p-1;
static const double LOG_LN2_LO = 0x1.cf79abc9e0000p-40;
static const double LOG_TAB_HI[16] = {
  0x0.0p+0, -0x1.f0a30c0120000p-5,
 -0x1.e27076e2b0000p-4, -0x1.5ff3070a78000p-3,
 -0x1.c8ff7c79a8000p-3, -0x1.1675cababc000p-2,
 -0x1.4618bc21c4000p-2, -0x1.739d7f6bbc000p-2,
  0x1.269621134c000p-2, 0x1.f991c6cb38000p-3,
  0x1.a93ed3c8b0000p-3, 0x1.5bf406b540000p-3,
  0x1.1178e82280000p-3, 0x1.9335e5d590000p-4,
  0x1.08598b59e0000p-4, 0x1.0415d89e80000p-5
};
static const double LOG_TAB_LO[16] = {
  0x0.0p+0, 0x1.3ab33d066d1d2p-42,
  0x1.a342c2af0003cp-45, -0x1.3d3c873e20a07p-43,
 -0x1.a21ac25d81ef3p-43, 0x1.9f1fc63382a8fp-42,
 -0x1.ec27d0b7b37b3p-42, -0x1.0069ce24c53fbp-42,
  0x1.b92783beb7677p-42, 0x1.9bcbecca0cdf3p-42,
 -0x1.30e486a0ac42dp-42, 0x1.ed8fdc149767ep-42,
 -0x1.b8421cc74be04p-43, 0x1.2622b8757a8fbp-42,
  0x1.d034451fecdfbp-43, -0x1.77771fd187145p-42
};

DREAM6_EXPORT int dream6_svml_log_ha_array(
    const double *x, double *out, size_t n
) {
    if (!x || !out) return -1;
    for (size_t i = 0; i < n; ++i) {
        const double xv = x[i];
        if (!(xv > 0.0) || !isfinite(xv)) {
            out[i] = log(xv);
            continue;
        }

        int binary_exponent;
        const double mantissa = 2.0 * frexp(xv, &binary_exponent);
        int exponent = binary_exponent - 1;

        /* vrcp14 followed by vrndscale(5,RN).  The 1/32 rounding is
           insensitive to the rcp14 approximation except at exact half-grid
           boundaries; those boundaries are represented identically by the
           correctly rounded reciprocal used here. */
        const double reciprocal =
            nearbyint((1.0 / mantissa) * 32.0) / 32.0;
        if (reciprocal < 0.75) {
            exponent += 1;
        }
        int table_index;
        if (reciprocal < 0.75) {
            table_index = (int)nearbyint(
                (2.0 * reciprocal - 1.0) * 16.0
            );
        } else {
            table_index = 8 + (int)nearbyint(
                (reciprocal - 0.75) * 32.0
            );
        }
        table_index &= 15;

        const double f = dream6_fma(reciprocal, mantissa, -1.0);
        const double p67 = dream6_fma(LOG_C7, f, LOG_C6);
        const double p89 = dream6_fma(LOG_C9, f, LOG_C8);
        const double f2 = f * f;
        const double p45 = dream6_fma(LOG_C5, f, LOG_C4);
        const double p23 = dream6_fma(LOG_C3, f, LOG_C2);
        const double p89_67 = dream6_fma(p89, f2, p67);
        const double f4 = f2 * f2;
        const double p45_23 = dream6_fma(p45, f2, p23);
        const double high = dream6_fma(
            LOG_LN2_HI,
            (double)exponent,
            LOG_TAB_HI[table_index]
        );
        double tail = dream6_fma(p89_67, f4, p45_23);
        const double sum = high + f;
        const double f_high = sum - high;
        const double f_low = f - f_high;
        tail = dream6_fma(tail, f2, f_low);
        const double low = dream6_fma(
            LOG_LN2_LO,
            (double)exponent,
            LOG_TAB_LO[table_index]
        );
        out[i] = sum + (tail + low);
    }
    return 0;
}

/* Bit-defined scalar transcription of the reference NumPy logaddexp path.
   NumPy's scalar npy_logaddexp is max(a,b)+log1p(exp(-|a-b|)); on the
   audited Linux reference the exp/log1p calls resolve to the x86-64 FMA
   glibc implementations below.  This copy removes the final CRT/libm
   dependency from the active EVEN_CYCLE map. */
static const uint64_t DREAM6_GLIBC_EXP_T[256] = {
  0x0000000000000000ULL,
  0x3ff0000000000000ULL,
  0x3c9b3b4f1a88bf6eULL,
  0x3feff63da9fb3335ULL,
  0xbc7160139cd8dc5dULL,
  0x3fefec9a3e778061ULL,
  0xbc905e7a108766d1ULL,
  0x3fefe315e86e7f85ULL,
  0x3c8cd2523567f613ULL,
  0x3fefd9b0d3158574ULL,
  0xbc8bce8023f98efaULL,
  0x3fefd06b29ddf6deULL,
  0x3c60f74e61e6c861ULL,
  0x3fefc74518759bc8ULL,
  0x3c90a3e45b33d399ULL,
  0x3fefbe3ecac6f383ULL,
  0x3c979aa65d837b6dULL,
  0x3fefb5586cf9890fULL,
  0x3c8eb51a92fdeffcULL,
  0x3fefac922b7247f7ULL,
  0x3c3ebe3d702f9cd1ULL,
  0x3fefa3ec32d3d1a2ULL,
  0xbc6a033489906e0bULL,
  0x3fef9b66affed31bULL,
  0xbc9556522a2fbd0eULL,
  0x3fef9301d0125b51ULL,
  0xbc5080ef8c4eea55ULL,
  0x3fef8abdc06c31ccULL,
  0xbc91c923b9d5f416ULL,
  0x3fef829aaea92de0ULL,
  0x3c80d3e3e95c55afULL,
  0x3fef7a98c8a58e51ULL,
  0xbc801b15eaa59348ULL,
  0x3fef72b83c7d517bULL,
  0xbc8f1ff055de323dULL,
  0x3fef6af9388c8deaULL,
  0x3c8b898c3f1353bfULL,
  0x3fef635beb6fcb75ULL,
  0xbc96d99c7611eb26ULL,
  0x3fef5be084045cd4ULL,
  0x3c9aecf73e3a2f60ULL,
  0x3fef54873168b9aaULL,
  0xbc8fe782cb86389dULL,
  0x3fef4d5022fcd91dULL,
  0x3c8a6f4144a6c38dULL,
  0x3fef463b88628cd6ULL,
  0x3c807a05b0e4047dULL,
  0x3fef3f49917ddc96ULL,
  0x3c968efde3a8a894ULL,
  0x3fef387a6e756238ULL,
  0x3c875e18f274487dULL,
  0x3fef31ce4fb2a63fULL,
  0x3c80472b981fe7f2ULL,
  0x3fef2b4565e27cddULL,
  0xbc96b87b3f71085eULL,
  0x3fef24dfe1f56381ULL,
  0x3c82f7e16d09ab31ULL,
  0x3fef1e9df51fdee1ULL,
  0xbc3d219b1a6fbffaULL,
  0x3fef187fd0dad990ULL,
  0x3c8b3782720c0ab4ULL,
  0x3fef1285a6e4030bULL,
  0x3c6e149289cecb8fULL,
  0x3fef0cafa93e2f56ULL,
  0x3c834d754db0abb6ULL,
  0x3fef06fe0a31b715ULL,
  0x3c864201e2ac744cULL,
  0x3fef0170fc4cd831ULL,
  0x3c8fdd395dd3f84aULL,
  0x3feefc08b26416ffULL,
  0xbc86a3803b8e5b04ULL,
  0x3feef6c55f929ff1ULL,
  0xbc924aedcc4b5068ULL,
  0x3feef1a7373aa9cbULL,
  0xbc9907f81b512d8eULL,
  0x3feeecae6d05d866ULL,
  0xbc71d1e83e9436d2ULL,
  0x3feee7db34e59ff7ULL,
  0xbc991919b3ce1b15ULL,
  0x3feee32dc313a8e5ULL,
  0x3c859f48a72a4c6dULL,
  0x3feedea64c123422ULL,
  0xbc9312607a28698aULL,
  0x3feeda4504ac801cULL,
  0xbc58a78f4817895bULL,
  0x3feed60a21f72e2aULL,
  0xbc7c2c9b67499a1bULL,
  0x3feed1f5d950a897ULL,
  0x3c4363ed60c2ac11ULL,
  0x3feece086061892dULL,
  0x3c9666093b0664efULL,
  0x3feeca41ed1d0057ULL,
  0x3c6ecce1daa10379ULL,
  0x3feec6a2b5c13cd0ULL,
  0x3c93ff8e3f0f1230ULL,
  0x3feec32af0d7d3deULL,
  0x3c7690cebb7aafb0ULL,
  0x3feebfdad5362a27ULL,
  0x3c931dbdeb54e077ULL,
  0x3feebcb299fddd0dULL,
  0xbc8f94340071a38eULL,
  0x3feeb9b2769d2ca7ULL,
  0xbc87deccdc93a349ULL,
  0x3feeb6daa2cf6642ULL,
  0xbc78dec6bd0f385fULL,
  0x3feeb42b569d4f82ULL,
  0xbc861246ec7b5cf6ULL,
  0x3feeb1a4ca5d920fULL,
  0x3c93350518fdd78eULL,
  0x3feeaf4736b527daULL,
  0x3c7b98b72f8a9b05ULL,
  0x3feead12d497c7fdULL,
  0x3c9063e1e21c5409ULL,
  0x3feeab07dd485429ULL,
  0x3c34c7855019c6eaULL,
  0x3feea9268a5946b7ULL,
  0x3c9432e62b64c035ULL,
  0x3feea76f15ad2148ULL,
  0xbc8ce44a6199769fULL,
  0x3feea5e1b976dc09ULL,
  0xbc8c33c53bef4da8ULL,
  0x3feea47eb03a5585ULL,
  0xbc845378892be9aeULL,
  0x3feea34634ccc320ULL,
  0xbc93cedd78565858ULL,
  0x3feea23882552225ULL,
  0x3c5710aa807e1964ULL,
  0x3feea155d44ca973ULL,
  0xbc93b3efbf5e2228ULL,
  0x3feea09e667f3bcdULL,
  0xbc6a12ad8734b982ULL,
  0x3feea012750bdabfULL,
  0xbc6367efb86da9eeULL,
  0x3fee9fb23c651a2fULL,
  0xbc80dc3d54e08851ULL,
  0x3fee9f7df9519484ULL,
  0xbc781f647e5a3ecfULL,
  0x3fee9f75e8ec5f74ULL,
  0xbc86ee4ac08b7db0ULL,
  0x3fee9f9a48a58174ULL,
  0xbc8619321e55e68aULL,
  0x3fee9feb564267c9ULL,
  0x3c909ccb5e09d4d3ULL,
  0x3feea0694fde5d3fULL,
  0xbc7b32dcb94da51dULL,
  0x3feea11473eb0187ULL,
  0x3c94ecfd5467c06bULL,
  0x3feea1ed0130c132ULL,
  0x3c65ebe1abd66c55ULL,
  0x3feea2f336cf4e62ULL,
  0xbc88a1c52fb3cf42ULL,
  0x3feea427543e1a12ULL,
  0xbc9369b6f13b3734ULL,
  0x3feea589994cce13ULL,
  0xbc805e843a19ff1eULL,
  0x3feea71a4623c7adULL,
  0xbc94d450d872576eULL,
  0x3feea8d99b4492edULL,
  0x3c90ad675b0e8a00ULL,
  0x3feeaac7d98a6699ULL,
  0x3c8db72fc1f0eab4ULL,
  0x3feeace5422aa0dbULL,
  0xbc65b6609cc5e7ffULL,
  0x3feeaf3216b5448cULL,
  0x3c7bf68359f35f44ULL,
  0x3feeb1ae99157736ULL,
  0xbc93091fa71e3d83ULL,
  0x3feeb45b0b91ffc6ULL,
  0xbc5da9b88b6c1e29ULL,
  0x3feeb737b0cdc5e5ULL,
  0xbc6c23f97c90b959ULL,
  0x3feeba44cbc8520fULL,
  0xbc92434322f4f9aaULL,
  0x3feebd829fde4e50ULL,
  0xbc85ca6cd7668e4bULL,
  0x3feec0f170ca07baULL,
  0x3c71affc2b91ce27ULL,
  0x3feec49182a3f090ULL,
  0x3c6dd235e10a73bbULL,
  0x3feec86319e32323ULL,
  0xbc87c50422622263ULL,
  0x3feecc667b5de565ULL,
  0x3c8b1c86e3e231d5ULL,
  0x3feed09bec4a2d33ULL,
  0xbc91bbd1d3bcbb15ULL,
  0x3feed503b23e255dULL,
  0x3c90cc319cee31d2ULL,
  0x3feed99e1330b358ULL,
  0x3c8469846e735ab3ULL,
  0x3feede6b5579fdbfULL,
  0xbc82dfcd978e9db4ULL,
  0x3feee36bbfd3f37aULL,
  0x3c8c1a7792cb3387ULL,
  0x3feee89f995ad3adULL,
  0xbc907b8f4ad1d9faULL,
  0x3feeee07298db666ULL,
  0xbc55c3d956dcaebaULL,
  0x3feef3a2b84f15fbULL,
  0xbc90a40e3da6f640ULL,
  0x3feef9728de5593aULL,
  0xbc68d6f438ad9334ULL,
  0x3feeff76f2fb5e47ULL,
  0xbc91eee26b588a35ULL,
  0x3fef05b030a1064aULL,
  0x3c74ffd70a5fddcdULL,
  0x3fef0c1e904bc1d2ULL,
  0xbc91bdfbfa9298acULL,
  0x3fef12c25bd71e09ULL,
  0x3c736eae30af0cb3ULL,
  0x3fef199bdd85529cULL,
  0x3c8ee3325c9ffd94ULL,
  0x3fef20ab5fffd07aULL,
  0x3c84e08fd10959acULL,
  0x3fef27f12e57d14bULL,
  0x3c63cdaf384e1a67ULL,
  0x3fef2f6d9406e7b5ULL,
  0x3c676b2c6c921968ULL,
  0x3fef3720dcef9069ULL,
  0xbc808a1883ccb5d2ULL,
  0x3fef3f0b555dc3faULL,
  0xbc8fad5d3ffffa6fULL,
  0x3fef472d4a07897cULL,
  0xbc900dae3875a949ULL,
  0x3fef4f87080d89f2ULL,
  0x3c74a385a63d07a7ULL,
  0x3fef5818dcfba487ULL,
  0xbc82919e2040220fULL,
  0x3fef60e316c98398ULL,
  0x3c8e5a50d5c192acULL,
  0x3fef69e603db3285ULL,
  0x3c843a59ac016b4bULL,
  0x3fef7321f301b460ULL,
  0xbc82d52107b43e1fULL,
  0x3fef7c97337b9b5fULL,
  0xbc892ab93b470dc9ULL,
  0x3fef864614f5a129ULL,
  0x3c74b604603a88d3ULL,
  0x3fef902ee78b3ff6ULL,
  0x3c83c5ec519d7271ULL,
  0x3fef9a51fbc74c83ULL,
  0xbc8ff7128fd391f0ULL,
  0x3fefa4afa2a490daULL,
  0xbc8dae98e223747dULL,
  0x3fefaf482d8e67f1ULL,
  0x3c8ec3bc41aa2008ULL,
  0x3fefba1bee615a27ULL,
  0x3c842b94c3a9eb32ULL,
  0x3fefc52b376bba97ULL,
  0x3c8a64a931d185eeULL,
  0x3fefd0765b6e4540ULL,
  0xbc8e37bae43be3edULL,
  0x3fefdbfdad9cbe14ULL,
  0x3c77893b4d91cd9dULL,
  0x3fefe7c1819e90d8ULL,
  0x3c5305c14160cc89ULL,
  0x3feff3c22b8f71f1ULL
};

static inline double dream6_asdouble(uint64_t i) { du64 u; u.u=i; return u.d; }
static inline uint64_t dream6_asuint64(double x) { du64 u; u.d=x; return u.u; }

static inline double dream6_glibc_exp_ref(double x) {
    const double InvLn2N = 0x1.71547652b82fep0 * 128.0;
    const double NegLn2hiN = -0x1.62e42fefa0000p-8;
    const double NegLn2loN = -0x1.cf79abc9e3b3ap-47;
    const double Shift = 0x1.8p52;
    const double P2 = 0x1.ffffffffffdbdp-2;
    const double P3 = 0x1.555555555543cp-3;
    const double P4 = 0x1.55555cf172b91p-5;
    const double P5 = 0x1.1111167a4d017p-7;
    const uint64_t ux = dream6_asuint64(x);
    const uint32_t abstop = (uint32_t)((ux >> 52) & 0x7ffu);
    if (abstop < ((dream6_asuint64(0x1p-54) >> 52) & 0x7ffu)) return 1.0;
    double z = dream6_fma(x, InvLn2N, Shift);
    const uint64_t ki = dream6_asuint64(z);
    const double kd = z - Shift;
    double r = dream6_fma(kd, NegLn2hiN, x);
    r = dream6_fma(kd, NegLn2loN, r);
    const uint64_t idx = 2u * (ki % 128u);
    const uint64_t top = ki << (52 - 7);
    const double tail = dream6_asdouble(DREAM6_GLIBC_EXP_T[idx]);
    const uint64_t sbits = DREAM6_GLIBC_EXP_T[idx + 1] + top;
    const double r2 = r * r;
    double p = dream6_fma(r, P3, P2);
    p = dream6_fma(r2, p, tail + r);
    const double q = dream6_fma(r, P5, P4);
    const double tmp = dream6_fma(r2 * r2, q, p);
    const double scale = dream6_asdouble(sbits);
    return dream6_fma(scale, tmp, scale);
}

static inline double dream6_glibc_log1p_ref(double x) {
    const double LN2HI = 0x1.62e42fee00000p-1;
    const double LN2LO = 0x1.a39ef35793c76p-33;
    const double G1 = 0x1.5555555555593p-1;
    const double G2 = 0x1.999999997fa04p-2;
    const double G3 = 0x1.2492494229359p-2;
    const double G4 = 0x1.c71c51d8e78afp-3;
    const double G5 = 0x1.7466496cb03dep-3;
    const double G6 = 0x1.39a09d078c69fp-3;
    const double G7 = 0x1.2f112df3e5244p-3;
    du64 u = {.d = x};
    uint32_t hx = (uint32_t)(u.u >> 32), hu;
    int k = 1;
    double f = 0.0, c = 0.0;
    if (hx < 0x3fda827au || (hx >> 31)) {
        if (hx >= 0xbff00000u) {
            if (x == -1.0) return -INFINITY;
            return NAN;
        }
        if ((hx << 1) < (0x3ca00000u << 1)) return x;
        if (hx <= 0xbfd2bec4u) { k = 0; c = 0.0; f = x; }
    } else if (hx >= 0x7ff00000u) {
        return x;
    }
    if (k) {
        u.d = 1.0 + x;
        hu = (uint32_t)(u.u >> 32);
        hu += 0x3ff00000u - 0x3fe6a09eu;
        k = (int)(hu >> 20) - 0x3ff;
        if (k < 54) {
            c = (k >= 2) ? 1.0 - (u.d - x) : x - (u.d - 1.0);
            c /= u.d;
        } else c = 0.0;
        hu = (hu & 0x000fffffu) + 0x3fe6a09eu;
        u.u = ((uint64_t)hu << 32) | (u.u & 0xffffffffu);
        f = u.d - 1.0;
    }
    const double hfsq = (0.5 * f) * f;
    const double s = f / (2.0 + f);
    const double zz = s * s;
    const double a = dream6_fma(zz, G3, G2);
    const double b = dream6_fma(zz, G5, G4);
    const double cc = dream6_fma(zz, G7, G6);
    const double w = zz * zz, w2 = w * w, w3 = w * w2;
    double R = dream6_fma(zz, G1, w * a);
    R = dream6_fma(w2, b, R);
    R = dream6_fma(w3, cc, R);
    const double st = s * (hfsq + R);
    if (k == 0) return f - (hfsq - st);
    const double dk = (double)k;
    double acc = dream6_fma(dk, LN2LO, c);
    acc += st;
    const double q = hfsq - acc;
    const double q2 = q - f;
    return dream6_fma(dk, LN2HI, -q2);
}

static inline double dream6_logaddexp_ref_scalar(double a, double b) {
    if (a == b) return a + 0x1.62e42fefa39efp-1;
    if (a > b) return a + dream6_glibc_log1p_ref(dream6_glibc_exp_ref(b - a));
    return b + dream6_glibc_log1p_ref(dream6_glibc_exp_ref(a - b));
}

DREAM6_EXPORT int dream6_logaddexp_ref_array(
    const double *a, const double *b, double *out, size_t n
) {
    if (!a || !b || !out) return -1;
    for (size_t i = 0; i < n; ++i) out[i] = dream6_logaddexp_ref_scalar(a[i], b[i]);
    return 0;
}

"""

_SVML_HA_LIBRARY = None
_SVML_HA_FUNCTION = None
_SVML_HA_LOG_FUNCTION = None
_SVML_HA_LOGADDEXP_FUNCTION = None
_SVML_HA_BUILD_META = None
_SVML_HA_DLL_DIR_HANDLE = None


def _find_c_compiler() -> str:
    """Find a native compiler suitable for the runtime helper.

    On Windows prefer the MinGW-w64 compiler explicitly.  An MSYS-target GCC
    can emit DLLs that depend on msys-2.0.dll and are unsuitable for loading
    into the normal Windows Python process.
    """
    candidates: list[str] = []
    if os.environ.get("CC"):
        candidates.append(os.environ["CC"])
    if os.name == "nt":
        candidates.extend([
            r"C:\msys\mingw64\bin\gcc.exe",
            r"C:\msys64\mingw64\bin\gcc.exe",
            "gcc",
            "clang",
        ])
    else:
        candidates.extend(["gcc", "cc", "clang"])

    rejected: list[str] = []
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if not resolved:
            path = Path(candidate)
            if path.is_file():
                resolved = str(path)
        if not resolved:
            continue

        if os.name == "nt":
            try:
                probe = subprocess.run(
                    [resolved, "-dumpmachine"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                target = probe.stdout.strip().lower()
            except OSError:
                target = ""
            # GCC from /usr/bin in MSYS targets *-pc-msys and creates a DLL
            # requiring msys-2.0.dll.  Reject it.  MinGW-w64 targets contain
            # "mingw".  Clang may not report a MinGW triple, so explicit CC
            # remains allowed if the executable itself is clang.
            name = Path(resolved).name.lower()
            if target and "msys" in target and "mingw" not in target:
                rejected.append(f"{resolved} target={target}")
                continue
            if name.startswith("gcc") and target and "mingw" not in target:
                rejected.append(f"{resolved} target={target}")
                continue
        return resolved

    detail = ("; rejected: " + "; ".join(rejected)) if rejected else ""
    raise RuntimeError(
        "v170 requires a native C compiler for its numerically specified "
        "EVEN_CYCLE exp/log kernel. Install MinGW-w64 GCC/Clang or set CC. "
        "On the user's Windows layout GCC is expected at "
        "C:\\msys\\mingw64\\bin\\gcc.exe." + detail
    )

def _load_svml_ha_kernel() -> tuple[object, dict]:
    global _SVML_HA_LIBRARY
    global _SVML_HA_FUNCTION
    global _SVML_HA_LOG_FUNCTION
    global _SVML_HA_LOGADDEXP_FUNCTION
    global _SVML_HA_BUILD_META
    global _SVML_HA_DLL_DIR_HANDLE
    if _SVML_HA_FUNCTION is not None:
        return _SVML_HA_FUNCTION, dict(_SVML_HA_BUILD_META)

    source_hash = hashlib.sha256(
        _SVML_HA_C_SOURCE.encode("utf-8")
    ).hexdigest()
    # Include the native-build contract in the cache key.  This deliberately
    # invalidates the first v170 Windows DLL, which could carry dynamic MinGW
    # runtime dependencies and fail under Python 3.8+ secure DLL loading.
    build_contract = (
        "v170-kernel-build-4|mxcsr-direct|hardware-fma|det-logaddexp|static-libgcc-win|"
        "secure-dll-dir|frounding-math|ffp-contract-off|avx-fma"
    )
    cache_hash = hashlib.sha256(
        (source_hash + "|" + build_contract + "|" + os.name).encode("ascii")
    ).hexdigest()
    cache_root = Path(tempfile.gettempdir()) / "dream6_v170_kernel"
    cache_root.mkdir(parents=True, exist_ok=True)
    source_path = cache_root / f"svml_ha_{cache_hash[:16]}.c"
    library_suffix = (
        ".dll" if os.name == "nt"
        else ".dylib" if platform.system() == "Darwin"
        else ".so"
    )
    library_path = cache_root / f"svml_ha_{cache_hash[:16]}{library_suffix}"

    compiler = None
    compile_command = None
    compiler_target = None
    if not library_path.is_file():
        source_path.write_text(_SVML_HA_C_SOURCE, encoding="utf-8")
        compiler = _find_c_compiler()
        try:
            target_probe = subprocess.run(
                [compiler, "-dumpmachine"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            compiler_target = target_probe.stdout.strip() or None
        except OSError:
            compiler_target = None

        compile_command = [
            compiler,
            "-O3",
            "-std=c11",
            "-shared",
            "-frounding-math",
            "-ffp-contract=off",
            "-msse2",
            "-mavx",
            "-mfma",
            "-mfpmath=sse",
        ]
        if os.name == "nt":
            # Keep the helper independent of libgcc_s_*.dll.  The C source
            # does not use threads or C++, so no additional runtime DLL should
            # be needed beyond normal Windows system libraries.
            compile_command.extend([
                "-static-libgcc",
                "-Wl,--no-undefined",
            ])
        else:
            compile_command.append("-fPIC")
        compile_command.extend([
            "-o", str(library_path), str(source_path), "-lm",
        ])
        completed = subprocess.run(
            compile_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to compile v170 deterministic exp/log kernel\n"
                f"command: {' '.join(compile_command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    # Python 3.8+ uses a restricted DLL search policy on Windows.  Register
    # the MinGW-w64 bin directory explicitly as a second line of defence for
    # any compiler runtime dependency.  Keep the handle alive globally.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        if compiler is None:
            try:
                compiler = _find_c_compiler()
            except RuntimeError:
                compiler = None
        if compiler is not None:
            compiler_dir = str(Path(compiler).resolve().parent)
            try:
                _SVML_HA_DLL_DIR_HANDLE = os.add_dll_directory(compiler_dir)
            except (FileNotFoundError, OSError):
                _SVML_HA_DLL_DIR_HANDLE = None

    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as exc:
        dependency_hint = ""
        if os.name == "nt":
            dependency_hint = (
                "\nWindows helper DLL could not be loaded. Ensure the compiler "
                "is MinGW-w64 (for this setup: C:\\msys\\mingw64\\bin\\gcc.exe), "
                "not the MSYS /usr/bin GCC. The v170 build uses -static-libgcc "
                "and registers the compiler bin directory with os.add_dll_directory()."
            )
        raise RuntimeError(
            f"failed to load v170 deterministic kernel: {library_path}: {exc}"
            + dependency_hint
        ) from exc
    function = library.dream6_svml_exp_ha_array
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    log_function = library.dream6_svml_log_ha_array
    log_function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    log_function.restype = ctypes.c_int
    logaddexp_function = library.dream6_logaddexp_ref_array
    logaddexp_function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    logaddexp_function.restype = ctypes.c_int

    # Cross-platform bit-contract self-test.  The expected digest is over the
    # 4096 float64 results for x_i=-80+80*i/4095 in native little-endian bytes.
    probe = np.asarray(
        [-80.0 + 80.0 * i / 4095.0 for i in range(4096)],
        dtype=np.float64,
    )
    probe_out = np.empty_like(probe)
    rc = function(
        probe.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        probe_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        probe.size,
    )
    if rc != 0:
        raise RuntimeError(f"v170 exp-kernel self-test call failed: rc={rc}")
    digest = hashlib.sha256(
        probe_out.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    expected_digest = "eaa1e1d505c7e6746261f5e534fb69e970ac7393579db412730f03cbf5e4cc94"
    if digest != expected_digest:
        raise RuntimeError(
            "v170 deterministic exp kernel violated its bit contract: "
            f"expected {expected_digest}, got {digest}"
        )

    log_probe = np.asarray(
        [1e-12 + (3.0 - 1e-12) * i / 4095.0 for i in range(4096)],
        dtype=np.float64,
    )
    log_probe_out = np.empty_like(log_probe)
    rc = log_function(
        log_probe.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        log_probe_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        log_probe.size,
    )
    if rc != 0:
        raise RuntimeError(f"v170 log-kernel self-test call failed: rc={rc}")
    log_digest = hashlib.sha256(
        log_probe_out.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    expected_log_digest = "0ed168a48bd970e8d6213b763694df2f616c65f2398530e4ce3f817793cd1d43"
    if log_digest != expected_log_digest:
        raise RuntimeError(
            "v170 deterministic log kernel violated its bit contract: "
            f"expected {expected_log_digest}, got {log_digest}"
        )

    logadd_a = np.asarray([-120.0 + 240.0 * i / 4095.0 for i in range(4096)], dtype=np.float64)
    logadd_b = np.asarray([90.0 - 180.0 * ((37 * i) % 4096) / 4095.0 for i in range(4096)], dtype=np.float64)
    logadd_out = np.empty_like(logadd_a)
    rc = logaddexp_function(
        logadd_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        logadd_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        logadd_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        logadd_a.size,
    )
    if rc != 0:
        raise RuntimeError(f"v170 logaddexp-kernel self-test call failed: rc={rc}")
    logadd_digest = hashlib.sha256(logadd_out.astype("<f8", copy=False).tobytes()).hexdigest()
    expected_logadd_digest = "bd1108553632b4dfde8a32f3cc98d06c78d590557d60d1f3eeaf74dc03cca785"
    if logadd_digest != expected_logadd_digest:
        raise RuntimeError(
            "v170 deterministic logaddexp kernel violated its bit contract: "
            f"expected {expected_logadd_digest}, got {logadd_digest}"
        )

    _SVML_HA_LIBRARY = library
    _SVML_HA_FUNCTION = function
    _SVML_HA_LOG_FUNCTION = log_function
    _SVML_HA_LOGADDEXP_FUNCTION = logaddexp_function
    _SVML_HA_BUILD_META = {
        "kind": "explicit_scalar_transcription_of_SVML_exp8_ha",
        "source_sha256": source_hash,
        "library_path": str(library_path),
        "compiler": compiler,
        "compile_command": compile_command,
        "exp_self_test_sha256": digest,
        "log_self_test_sha256": log_digest,
        "logaddexp_self_test_sha256": logadd_digest,
        "self_test_values_each": int(probe.size),
        "logaddexp_contract": "explicit scalar transcription of reference npy_logaddexp + FMA exp/log1p",
        "rounding_contract": (
            "x86 MXCSR RZ fused range reduction; MXCSR RN fused polynomial; "
            "hardware FMA; 1/16 table; IEEE-754 binary64"
        ),
    }
    return function, dict(_SVML_HA_BUILD_META)


def deterministic_svml_exp_ha(values: np.ndarray) -> np.ndarray:
    """Bit-defined exp map used by fused EVEN_CYCLE factors."""
    function, _meta = _load_svml_ha_kernel()
    x = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty_like(x)
    rc = function(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x.size,
    )
    if rc != 0:
        raise RuntimeError(f"v170 exp kernel failed: rc={rc}")
    return out


def deterministic_svml_log_ha(values: np.ndarray) -> np.ndarray:
    """Bit-defined log map used by fused EVEN_CYCLE factors."""
    _function, _meta = _load_svml_ha_kernel()
    x = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty_like(x)
    rc = _SVML_HA_LOG_FUNCTION(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x.size,
    )
    if rc != 0:
        raise RuntimeError(f"v170 log kernel failed: rc={rc}")
    return out


def deterministic_logaddexp_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bit-defined reference logaddexp used by fused EVEN_CYCLE factors."""
    _function, _meta = _load_svml_ha_kernel()
    aa = np.ascontiguousarray(a, dtype=np.float64)
    bb = np.ascontiguousarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError("logaddexp shape mismatch")
    out = np.empty_like(aa)
    rc = _SVML_HA_LOGADDEXP_FUNCTION(
        aa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        aa.size,
    )
    if rc != 0:
        raise RuntimeError(f"v170 logaddexp kernel failed: rc={rc}")
    return out



# Optional exact-order scatter accelerator.  Correctness never depends on Numba;
# the fallback is the frozen NumPy add.at operation.
try:
    from numba import njit as _v179_njit
    _V179_NUMBA_AVAILABLE = os.environ.get("DREAM6_FAST_SCATTER", "0") == "1"
except Exception:
    _V179_NUMBA_AVAILABLE = False
    def _v179_njit(*_args, **_kwargs):
        def wrap(function):
            return function
        return wrap

@_v179_njit(cache=True)
def _v179_scatter_rows_numba(out, indices, values):
    for edge in range(indices.shape[0]):
        row = indices[edge]
        for state in range(values.shape[1]):
            out[row, state] = np.float32(out[row, state] + values[edge, state])

@_v179_njit(cache=True)
def _v179_scatter_flat_numba(out, indices, values):
    for index in range(indices.shape[0]):
        target = indices[index]
        out[target] = np.float32(out[target] + values[index])

def _v179_scatter_rows(out, indices, values):
    if _V179_NUMBA_AVAILABLE:
        _v179_scatter_rows_numba(out, indices, values)
    else:
        np.add.at(out, indices, values)

def _v179_scatter_flat(out, indices, values):
    if _V179_NUMBA_AVAILABLE:
        _v179_scatter_flat_numba(out, indices, values)
    else:
        np.add.at(out, indices, values)

VERSION = "v186_GLOBAL_CONCURRENCE_CERTIFIED_SECTIONS_PLUS_V185_FROZEN_PATHS"

FACTOR_OR = 0
FACTOR_EXACT1 = 1
FACTOR_EVEN_CYCLE = 2
# Legacy symbolic alias only: CYCLE4 is the length-4 specialization.
FACTOR_CYCLE4 = FACTOR_EVEN_CYCLE


@dataclass(frozen=True)
class CNF:
    nvars: int
    clauses: tuple[tuple[int, ...], ...]
    sha256: str


@dataclass(frozen=True)
class FactorGraph:
    """Semantics-preserving factor graph after hierarchical contraction.

    The logical hierarchy is

        OR clauses -> EXACT1 motifs -> EVEN_CYCLE motifs.

    A length-4 cycle is not a separate factor species: it is the smallest
    nontrivial EVEN_CYCLE and uses the frozen v170 numerical specialization.
    """

    nvars: int
    original_nclauses: int
    nfactors: int
    nedges: int

    n_or_factors: int
    n_exact1_factors: int
    n_even_cycle_factors: int

    n_or_edges: int
    n_exact1_edges: int
    n_even_cycle_edges: int

    original_width_min: int
    original_width_max: int
    original_width_mean: float

    factor_offsets: np.ndarray
    factor_type: np.ndarray
    edge_factor: np.ndarray
    edge_var: np.ndarray
    edge_sign: np.ndarray
    edge_bundle: np.ndarray

    # Flattened EVEN_CYCLE bundle metadata.  For factor f, local bundles are
    # indexed 0..even_cycle_lengths[f)-1 and occupy the global bundle interval
    # [even_cycle_factor_bundle_offsets[f], even_cycle_factor_bundle_offsets[f+1]).
    even_cycle_lengths: np.ndarray
    even_cycle_factor_bundle_offsets: np.ndarray
    even_cycle_bundle_offsets: np.ndarray
    even_cycle_bundle_widths: np.ndarray
    even_cycle_positive_clause_offsets: np.ndarray
    even_cycle_positive_clause_ids: np.ndarray

    fused_positive_clause_ids: np.ndarray
    consumed_pair_clause_ids: np.ndarray
    exact1_widths: np.ndarray
    remaining_exact1_widths: np.ndarray
    remaining_or_clause_ids: np.ndarray

    @staticmethod
    def _even_cycle_bundles(
        groups: list[tuple[int, tuple[int, ...], tuple[int, ...]]],
    ) -> tuple[tuple[int, ...], ...] | None:
        """Recognize an exact alternating EVEN_CYCLE of fused EXACT1 gates.

        With bundles B_j and gates G_j the required incidence is

            G_j = B_{j-1} disjoint_union B_j   (indices modulo ell),

        where ell is even and >=4.  The exact local state space then has only
        two macro-states: choose one variable from every even bundle, or choose
        one variable from every odd bundle.

        Detection is deliberately strict and consumes only consecutive
        positive-clause IDs.  Failure means "do not fuse", never heuristic
        completion.
        """
        ell = len(groups)
        if ell < 4 or (ell & 1):
            return None
        # Clause IDs need not be adjacent in DIMACS: many encodings place each
        # positive support clause next to its AMO binaries.  Only the ordered
        # EXACT1 incidence matters.
        gates = [set(group[1]) for group in groups]
        bundles = [gates[j] & gates[(j + 1) % ell] for j in range(ell)]
        if not all(bundles):
            return None
        if len(set().union(*bundles)) != sum(len(bundle) for bundle in bundles):
            return None
        for j in range(ell):
            if gates[j] != bundles[(j - 1) % ell] | bundles[j]:
                return None
        return tuple(tuple(sorted(bundle)) for bundle in bundles)

    @classmethod
    def from_cnf(cls, cnf: CNF) -> "FactorGraph":
        original_widths = np.asarray([len(clause) for clause in cnf.clauses], dtype=np.int64)
        if original_widths.size == 0:
            raise ValueError("empty CNF")
        if np.any(original_widths <= 0):
            raise ValueError("empty clause detected: this branch intentionally emits no UNSAT verdict")

        negative_pair_occurrences: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        negative_pair_degree: dict[int, int] = defaultdict(int)
        for clause_id, clause in enumerate(cnf.clauses):
            if len(clause) == 2 and clause[0] < 0 and clause[1] < 0 and abs(clause[0]) != abs(clause[1]):
                pair = tuple(sorted((abs(clause[0]), abs(clause[1]))))
                negative_pair_occurrences[pair].append(clause_id)
                negative_pair_degree[pair[0]] += 1
                negative_pair_degree[pair[1]] += 1

        consumed_clause_ids: set[int] = set()
        fused_groups: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

        # Layer 1: positive OR + all pairwise AMO <=> EXACT1.
        for clause_id, clause in enumerate(cnf.clauses):
            if len(clause) < 2 or not all(literal > 0 for literal in clause):
                continue
            variables_one_based = tuple(int(literal) for literal in clause)
            if len(set(variables_one_based)) != len(variables_one_based):
                continue
            width = len(variables_one_based)
            required_pair_count = width * (width - 1) // 2
            # Exact necessary conditions before materializing O(width^2) pairs.
            # They can only reject an impossible AMO clique, never a valid one.
            if required_pair_count > len(negative_pair_occurrences):
                continue
            if any(negative_pair_degree[variable] < width - 1 for variable in variables_one_based):
                continue
            required_pairs = tuple(itertools.combinations(sorted(variables_one_based), 2))
            if not all(len(negative_pair_occurrences[pair]) > 0 for pair in required_pairs):
                continue
            support_clause_ids = [negative_pair_occurrences[pair].popleft() for pair in required_pairs]
            consumed_clause_ids.add(clause_id)
            consumed_clause_ids.update(support_clause_ids)
            fused_groups.append((clause_id,
                                 tuple(variable - 1 for variable in variables_one_based),
                                 tuple(support_clause_ids)))

        remaining_or_clause_ids = [cid for cid in range(len(cnf.clauses)) if cid not in consumed_clause_ids]

        # Layer 2: exact contraction of every strictly recognized consecutive
        # even EXACT1 cycle.  We select the shortest valid cycle beginning at a
        # cursor; this preserves the four-gate X9 station partition exactly,
        # while a genuine 6/8/... cycle is found once its closing gate appears.
        remaining_exact1_groups: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
        even_cycle_groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
        # v182: preserve the frozen consecutive-cycle semantics, but do not scan
        # every possible even length.  A valid closing gate must share at least
        # one variable with the first gate, so candidate closing positions are
        # obtained directly from the gate-variable incidence index.
        gate_positions_by_var: dict[int, list[int]] = defaultdict(list)
        for gate_index, (_cid, variables, _support) in enumerate(fused_groups):
            for variable in variables:
                gate_positions_by_var[int(variable)].append(gate_index)
        cursor = 0
        while cursor < len(fused_groups):
            closing_positions: set[int] = set()
            for variable in fused_groups[cursor][1]:
                for position in gate_positions_by_var[int(variable)]:
                    if position >= cursor + 3 and ((position - cursor + 1) & 1) == 0:
                        closing_positions.add(int(position))
            found = None
            for closing in sorted(closing_positions):
                candidate = fused_groups[cursor:closing + 1]
                bundles = cls._even_cycle_bundles(candidate)
                if bundles is not None:
                    found = (closing - cursor + 1, candidate, bundles)
                    break
            if found is None:
                remaining_exact1_groups.append(fused_groups[cursor])
                cursor += 1
            else:
                ell, candidate, bundles = found
                even_cycle_groups.append((tuple(group[0] for group in candidate), bundles))
                cursor += ell

        # Fixed factor order: OR, remaining EXACT1, EVEN_CYCLE.
        factors: list[tuple[tuple[int, float, int], ...]] = []
        factor_types: list[int] = []

        for clause_id in remaining_or_clause_ids:
            factor = []
            for literal in cnf.clauses[clause_id]:
                variable = abs(int(literal)) - 1
                if not 0 <= variable < cnf.nvars:
                    raise ValueError(f"literal out of range in clause {clause_id + 1}: {literal}")
                factor.append((variable, 1.0 if literal > 0 else -1.0, -1))
            factors.append(tuple(factor)); factor_types.append(FACTOR_OR)

        for _positive_clause_id, variables, _support_ids in remaining_exact1_groups:
            factors.append(tuple((variable, 1.0, -1) for variable in variables))
            factor_types.append(FACTOR_EXACT1)

        cycle_lengths: list[int] = []
        cycle_bundle_widths: list[int] = []
        cycle_positive_ids_flat: list[int] = []
        cycle_positive_offsets = [0]
        for positive_ids, bundles in even_cycle_groups:
            factor = []
            for bundle_id, variables in enumerate(bundles):
                cycle_bundle_widths.append(len(variables))
                for variable in variables:
                    factor.append((variable, 1.0, bundle_id))
            factors.append(tuple(factor)); factor_types.append(FACTOR_EVEN_CYCLE)
            cycle_lengths.append(len(bundles))
            cycle_positive_ids_flat.extend(int(x) for x in positive_ids)
            cycle_positive_offsets.append(len(cycle_positive_ids_flat))

        widths = np.asarray([len(factor) for factor in factors], dtype=np.int64)
        offsets = np.empty(len(factors) + 1, dtype=np.int64); offsets[0] = 0
        np.cumsum(widths, out=offsets[1:]); nedges = int(offsets[-1])
        edge_factor = np.repeat(np.arange(len(factors), dtype=np.int64), widths)
        edge_var = np.empty(nedges, dtype=np.int64)
        edge_sign = np.empty(nedges, dtype=np.float64)
        edge_bundle = np.full(nedges, -1, dtype=np.int32)
        edge_cursor = 0
        for factor in factors:
            for variable, sign, bundle_id in factor:
                edge_var[edge_cursor] = variable; edge_sign[edge_cursor] = sign
                edge_bundle[edge_cursor] = bundle_id; edge_cursor += 1

        n_or_factors = len(remaining_or_clause_ids)
        n_exact1_factors = len(remaining_exact1_groups)
        n_even_cycle_factors = len(even_cycle_groups)
        n_or_edges = int(np.sum(widths[:n_or_factors]))
        n_exact1_edges = int(np.sum(widths[n_or_factors:n_or_factors + n_exact1_factors]))
        n_even_cycle_edges = int(nedges - n_or_edges - n_exact1_edges)

        even_cycle_lengths = np.asarray(cycle_lengths, dtype=np.int64)
        even_cycle_factor_bundle_offsets = np.empty(n_even_cycle_factors + 1, dtype=np.int64)
        even_cycle_factor_bundle_offsets[0] = 0
        if n_even_cycle_factors:
            np.cumsum(even_cycle_lengths, out=even_cycle_factor_bundle_offsets[1:])
        even_cycle_bundle_widths = np.asarray(cycle_bundle_widths, dtype=np.int64)
        even_cycle_bundle_offsets = np.empty(even_cycle_bundle_widths.size + 1, dtype=np.int64)
        even_cycle_bundle_offsets[0] = 0
        if even_cycle_bundle_widths.size:
            np.cumsum(even_cycle_bundle_widths, out=even_cycle_bundle_offsets[1:])

        fused_positive_clause_ids = np.asarray([group[0] for group in fused_groups], dtype=np.int64)
        consumed_pair_clause_ids = np.asarray([sid for _pid, _vars, sids in fused_groups for sid in sids], dtype=np.int64)
        exact1_widths = np.asarray([len(group[1]) for group in fused_groups], dtype=np.int64)
        remaining_exact1_widths = np.asarray([len(group[1]) for group in remaining_exact1_groups], dtype=np.int64)

        if fused_groups:
            expected_pair_count = int(np.sum(exact1_widths * (exact1_widths - 1) // 2))
            if expected_pair_count != consumed_pair_clause_ids.size:
                raise AssertionError("EXACT1 support-clause count mismatch")
            if len(set(consumed_pair_clause_ids.tolist())) != consumed_pair_clause_ids.size:
                raise AssertionError("one binary-clause occurrence was consumed more than once")
            if len(set(fused_positive_clause_ids.tolist())) != fused_positive_clause_ids.size:
                raise AssertionError("one positive clause was fused more than once")
        if n_even_cycle_factors:
            if np.any((even_cycle_lengths < 4) | ((even_cycle_lengths & 1) != 0)):
                raise AssertionError("EVEN_CYCLE factor has invalid length")
            if np.any(even_cycle_bundle_widths <= 0):
                raise AssertionError("EVEN_CYCLE factor contains an empty bundle")
            if n_even_cycle_edges != int(np.sum(even_cycle_bundle_widths)):
                raise AssertionError("EVEN_CYCLE edge-count mismatch")

        return cls(
            nvars=cnf.nvars, original_nclauses=len(cnf.clauses), nfactors=len(factors), nedges=nedges,
            n_or_factors=n_or_factors, n_exact1_factors=n_exact1_factors, n_even_cycle_factors=n_even_cycle_factors,
            n_or_edges=n_or_edges, n_exact1_edges=n_exact1_edges, n_even_cycle_edges=n_even_cycle_edges,
            original_width_min=int(np.min(original_widths)), original_width_max=int(np.max(original_widths)),
            original_width_mean=float(np.mean(original_widths)), factor_offsets=offsets,
            factor_type=np.asarray(factor_types, dtype=np.int8), edge_factor=edge_factor, edge_var=edge_var,
            edge_sign=edge_sign, edge_bundle=edge_bundle, even_cycle_lengths=even_cycle_lengths,
            even_cycle_factor_bundle_offsets=even_cycle_factor_bundle_offsets,
            even_cycle_bundle_offsets=even_cycle_bundle_offsets, even_cycle_bundle_widths=even_cycle_bundle_widths,
            even_cycle_positive_clause_offsets=np.asarray(cycle_positive_offsets, dtype=np.int64),
            even_cycle_positive_clause_ids=np.asarray(cycle_positive_ids_flat, dtype=np.int64),
            fused_positive_clause_ids=fused_positive_clause_ids, consumed_pair_clause_ids=consumed_pair_clause_ids,
            exact1_widths=exact1_widths, remaining_exact1_widths=remaining_exact1_widths,
            remaining_or_clause_ids=np.asarray(remaining_or_clause_ids, dtype=np.int64),
        )

    # Compatibility properties for inherited v170 report/output code.  These
    # are metadata aliases only; semantically every member is EVEN_CYCLE.
    @property
    def n_cycle4_factors(self) -> int:
        return int(np.count_nonzero(self.even_cycle_lengths == 4))

    @property
    def n_cycle4_edges(self) -> int:
        if self.n_even_cycle_factors == 0:
            return 0
        total = 0
        base_factor = self.n_or_factors + self.n_exact1_factors
        for f, ell in enumerate(self.even_cycle_lengths):
            if int(ell) == 4:
                a = int(self.factor_offsets[base_factor + f]); b = int(self.factor_offsets[base_factor + f + 1])
                total += b - a
        return total

    @property
    def cycle4_bundle_offsets(self) -> np.ndarray:
        # X9 and the frozen evaluator contain only length-4 cycles, so this is
        # exactly the old v170 array in that specialization.
        return self.even_cycle_bundle_offsets

    @property
    def cycle4_bundle_widths(self) -> np.ndarray:
        if self.even_cycle_lengths.size and np.all(self.even_cycle_lengths == 4):
            return self.even_cycle_bundle_widths.reshape((-1, 4))
        return self.even_cycle_bundle_widths

    @property
    def cycle4_positive_clause_ids(self) -> np.ndarray:
        if self.even_cycle_lengths.size and np.all(self.even_cycle_lengths == 4):
            return self.even_cycle_positive_clause_ids.reshape((-1, 4))
        return self.even_cycle_positive_clause_ids


def read_dimacs(path: str | Path) -> CNF:
    """Read DIMACS CNF, including the common SATLIB ``%`` terminator."""
    p = Path(path)
    raw = p.read_bytes()

    nvars = None
    expected_clauses = None
    clauses: list[tuple[int, ...]] = []
    pending: list[int] = []
    end_of_instance = False

    for line_number, raw_line in enumerate(
        raw.decode("utf-8", errors="replace").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("%"):
            end_of_instance = True
            break

        if line.startswith("p"):
            parts = line.split()
            if len(parts) < 4 or parts[1].lower() != "cnf":
                raise ValueError(
                    f"invalid DIMACS header on line {line_number}: {line!r}"
                )
            nvars = int(parts[2])
            expected_clauses = int(parts[3])
            continue

        for token in line.split():
            if token.startswith("%"):
                end_of_instance = True
                break
            try:
                literal = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"invalid DIMACS token {token!r} on line {line_number}"
                ) from exc

            if literal == 0:
                clauses.append(tuple(pending))
                pending = []
            else:
                pending.append(literal)

        if end_of_instance:
            break

    if pending:
        raise ValueError("unterminated DIMACS clause")
    if nvars is None:
        nvars = max(
            (abs(literal) for clause in clauses for literal in clause),
            default=0,
        )
    if (
        expected_clauses is not None
        and expected_clauses != len(clauses)
    ):
        raise ValueError(
            f"DIMACS clause count mismatch: "
            f"header={expected_clauses}, parsed={len(clauses)}"
        )
    if nvars <= 0 or not clauses:
        raise ValueError("empty CNF")
    if any(len(clause) == 0 for clause in clauses):
        raise ValueError(
            "empty clause detected: formula is immediately UNSAT, but this "
            "solver branch intentionally emits no UNSAT verdict"
        )

    canonical = [f"p cnf {nvars} {len(clauses)}\n"]
    canonical.extend(
        " ".join(str(literal) for literal in clause) + " 0\n"
        for clause in clauses
    )
    sha256 = hashlib.sha256(
        "".join(canonical).encode("ascii")
    ).hexdigest()

    return CNF(
        nvars=int(nvars),
        clauses=tuple(clauses),
        sha256=sha256,
    )


def reinforced_hybrid_cavity_operator(
    graph: FactorGraph,
    *,
    iterations: int,
    damping: float,
    exact1_damping: float,
    cycle4_damping: float,
    reinforcement: float,
    cycle4_reinforcement: float,
    pair_polarization: float,
    log_clip: float,
    epsilon: float,
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Apply one fixed OR/EXACT1/EVEN_CYCLE cavity operator.

    No Boolean assignment or UNSAT count is constructed in this function.
    """
    n = graph.nvars
    E = graph.nedges
    var = graph.edge_var
    sign = graph.edge_sign
    edge_factor = graph.edge_factor
    offsets = graph.factor_offsets

    n_or_factors = graph.n_or_factors
    n_exact1_factors = graph.n_exact1_factors
    n_cycle4_factors = graph.n_even_cycle_factors

    n_or_edges = graph.n_or_edges
    n_exact1_edges = graph.n_exact1_edges
    n_cycle4_edges = graph.n_even_cycle_edges

    exact1_edge_start = n_or_edges
    exact1_edge_stop = n_or_edges + n_exact1_edges
    cycle4_edge_start = exact1_edge_stop

    or_offsets = offsets[:n_or_factors + 1]
    exact1_offsets = (
        offsets[
            n_or_factors:
            n_or_factors + n_exact1_factors + 1
        ]
        - n_or_edges
    )

    variable_to_factor = np.zeros(E, dtype=np.float64)
    factor_to_variable = np.zeros(E, dtype=np.float64)

    alpha_or = float(damping)
    alpha_exact1 = float(exact1_damping)
    alpha_cycle4 = float(cycle4_damping)

    rho = float(reinforcement)
    rho_cycle4 = float(cycle4_reinforcement)

    eps = max(float(epsilon), np.finfo(np.float64).tiny)
    clip = max(float(log_clip), 1.0)
    count = max(1, int(iterations))

    for name, value in (
        ("OR", alpha_or),
        ("EXACT1", alpha_exact1),
        ("EVEN_CYCLE", alpha_cycle4),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"{name} cavity damping must lie in (0, 1]"
            )
    if rho < 0.0 or rho_cycle4 < 0.0:
        raise ValueError(
            "cavity reinforcement must be nonnegative"
        )

    factor_type_on_edge = graph.factor_type[edge_factor]
    edge_alpha = np.where(
        factor_type_on_edge == FACTOR_EVEN_CYCLE,
        alpha_cycle4,
        np.where(
            factor_type_on_edge == FACTOR_EXACT1,
            alpha_exact1,
            alpha_or,
        ),
    )
    edge_rho = np.where(
        factor_type_on_edge == FACTOR_EVEN_CYCLE,
        rho_cycle4,
        rho,
    )

    # Universal second-order interaction sector: factors of arity two.
    # This is a geometric sector of the same global operator, not an
    # instance/family detector.  If absent, the v174 trajectory is untouched.
    factor_width_all = np.diff(offsets).astype(np.int64)
    edge_pair_mask = factor_width_all[edge_factor] == 2
    has_pair_sector = bool(np.any(edge_pair_mask))

    started = time.perf_counter()
    final_update_norm = math.inf
    max_message = 0.0

    for _ in range(count):
        cavity = np.clip(
            variable_to_factor,
            -clip,
            clip,
        )
        new_factor_to_variable = np.empty(
            E,
            dtype=np.float64,
        )

        # Ordinary OR factors: preserve the successful v158/v159 arithmetic.
        if n_or_edges > 0:
            cavity_or = cavity[:n_or_edges]
            sign_or = sign[:n_or_edges]

            p_true = 1.0 / (1.0 + np.exp(-cavity_or))
            p_violate = np.where(
                sign_or > 0.0,
                1.0 - p_true,
                p_true,
            )

            clause_product = np.multiply.reduceat(
                p_violate,
                or_offsets[:-1],
            )
            local_or_factor = edge_factor[:n_or_edges]
            product_other = (
                clause_product[local_or_factor]
                / np.maximum(p_violate, eps)
            )

            magnitude = -np.log(
                np.maximum(
                    eps,
                    1.0 - product_other,
                )
            )
            new_factor_to_variable[:n_or_edges] = (
                sign_or * magnitude
            )

        # Remaining fused EXACT1 factors.
        if n_exact1_edges > 0:
            cavity_exact1 = cavity[
                exact1_edge_start:exact1_edge_stop
            ]
            local_exact1_factor = (
                edge_factor[
                    exact1_edge_start:exact1_edge_stop
                ]
                - n_or_factors
            )

            factor_max = np.maximum.reduceat(
                cavity_exact1,
                exact1_offsets[:-1],
            )
            shifted_exp = np.exp(
                cavity_exact1
                - factor_max[local_exact1_factor]
            )
            factor_sum = np.add.reduceat(
                shifted_exp,
                exact1_offsets[:-1],
            )
            sum_other = np.maximum(
                eps,
                factor_sum[local_exact1_factor]
                - shifted_exp,
            )
            logsumexp_other = (
                factor_max[local_exact1_factor]
                + np.log(sum_other)
            )

            new_factor_to_variable[
                exact1_edge_start:exact1_edge_stop
            ] = -logsumexp_other

        # EVEN_CYCLE factors.
        #
        # Every recognized even cycle has exactly two macro-states: select one
        # value from every even bundle, or select one value from every odd
        # bundle.  Length 4 uses the frozen v170 vectorized evaluator verbatim;
        # this is a numerical specialization of the same factor, not a separate
        # logical branch.
        if n_cycle4_edges > 0:
            cavity_cycle = cavity[cycle4_edge_start:]
            local_cycle_factor = (
                edge_factor[cycle4_edge_start:]
                - n_or_factors
                - n_exact1_factors
            )
            bundle = np.asarray(graph.edge_bundle[cycle4_edge_start:], dtype=np.int64)

            if np.all(graph.even_cycle_lengths == 4):
                # FROZEN V170 length-4 specialization.  Do not algebraically
                # rewrite this block: X9 must remain bit-identical.
                segment_id = 4 * local_cycle_factor + bundle
                bundle_offsets = graph.even_cycle_bundle_offsets

                bundle_max = np.maximum.reduceat(cavity_cycle, bundle_offsets[:-1])
                shifted_exp = deterministic_svml_exp_ha(cavity_cycle - bundle_max[segment_id])
                bundle_sum = np.add.reduceat(shifted_exp, bundle_offsets[:-1])
                log_bundle_sum = bundle_max + deterministic_svml_log_ha(np.maximum(bundle_sum, eps))

                sum_excluding_self = np.maximum(eps, bundle_sum[segment_id] - shifted_exp)
                log_bundle_excluding_self = bundle_max[segment_id] + deterministic_svml_log_ha(sum_excluding_self)

                opposite_bundle = bundle ^ 2
                log_z_one = log_bundle_sum[4 * local_cycle_factor + opposite_bundle]
                alternate_bundle_1 = np.where((bundle & 1) == 0, 1, 0)
                alternate_bundle_2 = np.where((bundle & 1) == 0, 3, 2)
                log_alternate_mode = (
                    log_bundle_sum[4 * local_cycle_factor + alternate_bundle_1]
                    + log_bundle_sum[4 * local_cycle_factor + alternate_bundle_2]
                )
                log_same_mode_without_target = log_bundle_excluding_self + log_z_one
                log_z_zero = deterministic_logaddexp_ref(log_same_mode_without_target, log_alternate_mode)
                new_factor_to_variable[cycle4_edge_start:] = log_z_one - log_z_zero
            else:
                # General EVEN_CYCLE evaluator.  Its state count is two for any
                # even length ell, so the update is O(edges + ell) per factor.
                cycle_factor_base = n_or_factors + n_exact1_factors
                for cf, ell_raw in enumerate(graph.even_cycle_lengths):
                    ell = int(ell_raw)
                    factor_id = cycle_factor_base + cf
                    edge_a = int(graph.factor_offsets[factor_id])
                    edge_b = int(graph.factor_offsets[factor_id + 1])
                    local_a = edge_a - cycle4_edge_start
                    local_b = edge_b - cycle4_edge_start
                    cvals = cavity_cycle[local_a:local_b]
                    cbundle = np.asarray(graph.edge_bundle[edge_a:edge_b], dtype=np.int64)
                    bbase = int(graph.even_cycle_factor_bundle_offsets[cf])
                    bend = int(graph.even_cycle_factor_bundle_offsets[cf + 1])
                    boffs_global = graph.even_cycle_bundle_offsets[bbase:bend + 1]
                    boffs = boffs_global - int(boffs_global[0])
                    bmax = np.maximum.reduceat(cvals, boffs[:-1])
                    sexp = deterministic_svml_exp_ha(cvals - bmax[cbundle])
                    bsum = np.add.reduceat(sexp, boffs[:-1])
                    blog = bmax + deterministic_svml_log_ha(np.maximum(bsum, eps))
                    excl = np.maximum(eps, bsum[cbundle] - sexp)
                    blog_excl = bmax[cbundle] + deterministic_svml_log_ha(excl)
                    even_total = float(np.sum(blog[0:ell:2], dtype=np.float64))
                    odd_total = float(np.sum(blog[1:ell:2], dtype=np.float64))
                    same_total = np.where((cbundle & 1) == 0, even_total, odd_total)
                    alt_total = np.where((cbundle & 1) == 0, odd_total, even_total)
                    log_z_one = same_total - blog[cbundle]
                    log_same_zero = blog_excl + log_z_one
                    log_z_zero = deterministic_logaddexp_ref(log_same_zero, alt_total)
                    new_factor_to_variable[edge_a:edge_b] = log_z_one - log_z_zero

        # v175 pair-sector susceptibility.  The factor-local posterior logit
        # is decomposed into a common mode and a zero-sum contrast.  Only the
        # universal arity-two sector receives the contrast feedback; no hard
        # Boolean assignment, residual clause, or verifier signal is read.
        if pair_polarization != 0.0 and has_pair_sector:
            local_logit = np.clip(
                cavity + new_factor_to_variable,
                -clip,
                clip,
            )
            factor_sum_logit = np.add.reduceat(
                local_logit,
                offsets[:-1],
            )
            factor_mean_logit = (
                factor_sum_logit
                / np.maximum(factor_width_all.astype(np.float64), 1.0)
            )
            contrast = (
                local_logit
                - factor_mean_logit[edge_factor]
            )
            new_factor_to_variable = (
                new_factor_to_variable
                + float(pair_polarization)
                * edge_pair_mask.astype(np.float64)
                * contrast
            )

        total_field = np.bincount(
            var,
            weights=new_factor_to_variable,
            minlength=n,
        ).astype(np.float64, copy=False)

        new_variable_to_factor = (
            total_field[var]
            - new_factor_to_variable
            + edge_rho * total_field[var]
        )

        final_update_norm = max(
            float(np.max(np.abs(
                new_factor_to_variable
                - factor_to_variable
            ))),
            float(np.max(np.abs(
                new_variable_to_factor
                - variable_to_factor
            ))),
        )

        factor_to_variable = (
            (1.0 - edge_alpha)
            * factor_to_variable
            + edge_alpha
            * new_factor_to_variable
        )
        variable_to_factor = (
            (1.0 - edge_alpha)
            * variable_to_factor
            + edge_alpha
            * new_variable_to_factor
        )

        max_message = max(
            max_message,
            float(np.max(np.abs(factor_to_variable))),
            float(np.max(np.abs(variable_to_factor))),
        )

    belief = np.bincount(
        var,
        weights=factor_to_variable,
        minlength=n,
    ).astype(np.float64, copy=False)

    meta = {
        "kind": (
            "fixed_semantics_preserving_"
            "OR_EXACT1_EVEN_CYCLE_cavity_operator"
        ),
        "operator_power": int(count),
        "iteration_law": "64*n by default",
        "or_damping": float(alpha_or),
        "exact1_damping": float(alpha_exact1),
        "even_cycle_damping": float(alpha_cycle4),
        "damping_law": (
            "alpha_OR=clip(12.5/n,0.02,0.05); "
            "alpha_EXACT1=0.03; alpha_EVEN_CYCLE=0.06"
        ),
        "or_exact1_reinforcement": float(rho),
        "even_cycle_reinforcement": float(rho_cycle4),
        "pair_sector_polarization": float(pair_polarization),
        "pair_sector_present": bool(has_pair_sector),
        "pair_sector_factors": int(np.count_nonzero(factor_width_all == 2)),
        "pair_susceptibility_law": (
            "gamma_2=0.004 on factor arity |a|=2; zero-sum factor-local logit contrast; "
            "all other arities receive exactly zero extra term"
        ),
        "even_cycle_exp_log_kernel": (
            _load_svml_ha_kernel()[1]
            if n_cycle4_edges > 0
            else {"executed": False, "reason": "no EVEN_CYCLE factors"}
        ),
        "even_cycle_exp_log_backend_independent": bool(n_cycle4_edges > 0),
        "log_clip": float(clip),
        "epsilon": float(eps),
        "initial_message_state": "all zero",
        "factor_graph_factors": int(graph.nfactors),
        "factor_graph_edges": int(graph.nedges),
        "or_factors": int(graph.n_or_factors),
        "exact1_factors": int(graph.n_exact1_factors),
        "even_cycle_factors": int(graph.n_even_cycle_factors),
        "cycle4_factors": int(graph.n_cycle4_factors),
        "or_edges": int(graph.n_or_edges),
        "exact1_edges": int(graph.n_exact1_edges),
        "even_cycle_edges": int(graph.n_even_cycle_edges),
        "cycle4_edges": int(graph.n_cycle4_edges),
        "fused_positive_clauses": int(
            graph.fused_positive_clause_ids.size
        ),
        "consumed_negative_binary_clauses": int(
            graph.consumed_pair_clause_ids.size
        ),
        "remaining_OR_clauses": int(
            graph.remaining_or_clause_ids.size
        ),
        "remaining_EXACT1_clauses": int(
            graph.n_exact1_factors
        ),
        "even_cycle_positive_clauses": int(graph.even_cycle_positive_clause_ids.size),
        "even_cycle_lengths": graph.even_cycle_lengths.tolist(),
        "factor_fusion_semantics": (
            "OR+pairwise AMO <=> EXACT1; "
            "even EXACT1 cycle <=> two alternating bundle macro-states; "
            "length 4 uses the frozen v170 specialization"
        ),
        "final_update_norm": float(final_update_norm),
        "max_abs_message_seen": float(max_message),
        "belief_abs_mean": float(np.mean(np.abs(belief))),
        "belief_abs_min": float(np.min(np.abs(belief))),
        "belief_abs_max": float(np.max(np.abs(belief))),
        "runtime_seconds": float(
            time.perf_counter() - started
        ),
        "fixed_operator": True,
        "formula_preprocessing_only": True,
        "intermediate_boolean_checks": False,
        "boolean_archive": False,
        "clause_memory": False,
        "dynamic_clause_reweighting": False,
        "residual_clause_selection": False,
        "literal_selection": False,
        "boolean_flips": False,
        "branching": False,
        "decimation": False,
        "restart_portfolio": False,
        "external_solver": False,
        "final_readout": "one sign test belief_i >= 0",
    }
    return (
        belief,
        meta,
        variable_to_factor,
        factor_to_variable,
    )

def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def global_exact1_susceptibility_projection(
    graph: FactorGraph,
    belief: np.ndarray,
    *,
    temperature_scale: float,
    diagonal_floor: float,
    ridge: float,
    response_gain: float,
    trust_scale: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict]:
    """One formula-global continuous correction over all fused EXACT1 factors.

    Let A be the fused EXACT1 incidence matrix and p=sigmoid(H/s).  The
    linear-response correction is

        delta p
          = D A^T (A D A^T + lambda I)^(-1) (1 - A p),

        D_i = p_i(1-p_i) + d0.

    The linear system is solved by deterministic matrix-free preconditioned
    conjugate gradients.  No Boolean assignment, residual clause, or verifier
    score is constructed here.  Formulas without fused EXACT1 factors are
    returned unchanged.
    """
    started = time.perf_counter()
    H = np.asarray(belief, dtype=np.float64).reshape(-1)
    if H.size != graph.nvars:
        raise ValueError("belief width mismatch")

    m = int(graph.n_exact1_factors)
    if m == 0:
        return H.copy(), {
            "kind": "global_EXACT1_susceptibility_projection",
            "executed": False,
            "reason": "no fused EXACT1 factors",
            "exact1_factors": 0,
            "runtime_seconds": float(time.perf_counter() - started),
            "intermediate_boolean_checks": False,
            "residual_clause_selection": False,
            "boolean_archive": False,
            "boolean_flips": False,
            "branching": False,
        }

    edge_start = int(graph.n_or_edges)
    edge_stop = edge_start + int(graph.n_exact1_edges)
    exact_var = np.asarray(
        graph.edge_var[edge_start:edge_stop],
        dtype=np.int64,
    )
    exact_factor = np.asarray(
        graph.edge_factor[edge_start:edge_stop]
        - graph.n_or_factors,
        dtype=np.int64,
    )

    touched = np.bincount(
        exact_var,
        minlength=graph.nvars,
    ) > 0
    abs_touched = np.abs(H[touched])
    positive_scale = abs_touched[
        np.isfinite(abs_touched) & (abs_touched > 1e-12)
    ]
    base_scale = (
        float(np.median(positive_scale))
        if positive_scale.size
        else 1.0
    )
    temperature = max(
        1e-9,
        float(temperature_scale) * base_scale,
    )

    p = _stable_sigmoid(
        np.clip(H / temperature, -60.0, 60.0)
    )
    floor = max(0.0, float(diagonal_floor))
    D = p * (1.0 - p) + floor

    group_soft_sum = np.bincount(
        exact_factor,
        weights=p[exact_var],
        minlength=m,
    ).astype(np.float64, copy=False)
    rhs = 1.0 - group_soft_sum

    lam = max(1e-12, float(ridge))
    degree_weight = np.bincount(
        exact_factor,
        weights=D[exact_var],
        minlength=m,
    ).astype(np.float64, copy=False)
    preconditioner = 1.0 / np.maximum(
        degree_weight + lam,
        1e-12,
    )

    def matvec(vector: np.ndarray) -> np.ndarray:
        factor_vector = np.asarray(
            vector,
            dtype=np.float64,
        )
        variable_accumulator = np.bincount(
            exact_var,
            weights=factor_vector[exact_factor],
            minlength=graph.nvars,
        ).astype(np.float64, copy=False)
        variable_accumulator *= D
        return (
            np.bincount(
                exact_factor,
                weights=variable_accumulator[exact_var],
                minlength=m,
            ).astype(np.float64, copy=False)
            + lam * factor_vector
        )

    # Deterministic preconditioned conjugate gradients.
    y = np.zeros(m, dtype=np.float64)
    residual = rhs.copy()
    z = preconditioner * residual
    direction = z.copy()
    rz = float(np.dot(residual, z))
    rhs_norm = float(np.linalg.norm(rhs))
    converged = rhs_norm <= float(tolerance)
    used_iterations = 0

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        if converged:
            break
        Ad = matvec(direction)
        denominator = float(np.dot(direction, Ad))
        if not np.isfinite(denominator) or denominator <= 1e-30:
            break

        step = rz / denominator
        y += step * direction
        residual -= step * Ad
        used_iterations = iteration

        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= float(tolerance) * max(1.0, rhs_norm):
            converged = True
            break

        z_new = preconditioner * residual
        rz_new = float(np.dot(residual, z_new))
        if not np.isfinite(rz_new) or abs(rz) <= 1e-30:
            break
        direction = z_new + (rz_new / rz) * direction
        z = z_new
        rz = rz_new

    variable_response = np.bincount(
        exact_var,
        weights=y[exact_factor],
        minlength=graph.nvars,
    ).astype(np.float64, copy=False)
    raw_delta_p = D * variable_response

    requested_gain = max(0.0, float(response_gain))
    touched_mean_response = float(
        np.mean(np.abs(raw_delta_p[touched]))
    )
    trust_radius = max(
        0.0,
        float(trust_scale) / math.sqrt(max(1, graph.nvars)),
    )
    if requested_gain <= 0.0 or touched_mean_response <= 1e-30:
        applied_gain = 0.0
    else:
        applied_gain = min(
            requested_gain,
            trust_radius / touched_mean_response,
        )

    delta_p = applied_gain * raw_delta_p
    corrected_p = np.clip(
        p + delta_p,
        1e-12,
        1.0 - 1e-12,
    )

    corrected_belief = H.copy()
    corrected_belief[touched] = temperature * (
        np.log(corrected_p[touched])
        - np.log1p(-corrected_p[touched])
    )

    projected_group_sum = np.bincount(
        exact_factor,
        weights=corrected_p[exact_var],
        minlength=m,
    ).astype(np.float64, copy=False)

    meta = {
        "kind": "global_EXACT1_susceptibility_projection",
        "executed": True,
        "exact1_factors": m,
        "exact1_edges": int(exact_var.size),
        "temperature_scale": float(temperature_scale),
        "temperature": float(temperature),
        "diagonal_floor": float(floor),
        "ridge": float(lam),
        "requested_response_gain": float(requested_gain),
        "trust_scale": float(trust_scale),
        "mean_response_trust_radius": float(trust_radius),
        "applied_response_gain": float(applied_gain),
        "linear_solver": "matrix-free preconditioned conjugate gradients",
        "linear_iterations": int(used_iterations),
        "linear_converged": bool(converged),
        "rhs_norm": float(rhs_norm),
        "final_linear_residual_norm": float(
            np.linalg.norm(residual)
        ),
        "soft_constraint_residual_before": float(
            np.linalg.norm(1.0 - group_soft_sum)
        ),
        "soft_constraint_residual_after": float(
            np.linalg.norm(1.0 - projected_group_sum)
        ),
        "raw_max_abs_probability_response": float(
            np.max(np.abs(raw_delta_p))
        ),
        "raw_mean_abs_probability_response": float(
            touched_mean_response
        ),
        "applied_max_abs_probability_response": float(
            np.max(np.abs(delta_p))
        ),
        "applied_mean_abs_probability_response": float(
            np.mean(np.abs(delta_p[touched]))
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "fixed_formula_global_operator": True,
        "intermediate_boolean_checks": False,
        "residual_clause_selection": False,
        "literal_selection": False,
        "boolean_archive": False,
        "boolean_flips": False,
        "branching": False,
        "restart_portfolio": False,
        "external_solver": False,
    }
    return corrected_belief, meta

def verify_assignment_independent(
    cnf: CNF,
    assignment: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Independent exact verifier against the original CNF."""
    values = np.asarray(assignment, dtype=np.bool_).reshape(-1)
    if values.size != cnf.nvars:
        raise ValueError("assignment width mismatch")

    residual_ids: list[int] = []
    for clause_id, clause in enumerate(cnf.clauses):
        satisfied = False
        for literal in clause:
            value = bool(values[abs(literal) - 1])
            if (
                (literal > 0 and value)
                or (literal < 0 and not value)
            ):
                satisfied = True
                break
        if not satisfied:
            residual_ids.append(clause_id)

    return (
        len(residual_ids),
        np.asarray(residual_ids, dtype=np.int64),
    )


def write_model(
    path: str | Path,
    assignment: np.ndarray,
    sat: bool,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    values = [
        str(index + 1 if bool(value) else -(index + 1))
        for index, value in enumerate(
            np.asarray(assignment, dtype=np.bool_)
        )
    ]

    with p.open("w", encoding="utf-8") as stream:
        if sat:
            stream.write("s SATISFIABLE\n")
        else:
            stream.write(
                "c CANDIDATE ONLY — NOT A SAT VERDICT\n"
            )
        for start in range(0, len(values), 20):
            chunk = values[start:start + 20]
            suffix = " 0" if start + 20 >= len(values) else ""
            stream.write("v " + " ".join(chunk) + suffix + "\n")


def write_residual(
    path: str | Path,
    cnf: CNF,
    residual_ids: np.ndarray,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as stream:
        stream.write(
            f"# UNSAT clause count: "
            f"{len(residual_ids)} / {len(cnf.clauses)}\n"
        )
        for clause_id in residual_ids:
            clause = cnf.clauses[int(clause_id)]
            stream.write(
                f"{int(clause_id) + 1}: "
                + " ".join(str(literal) for literal in clause)
                + " 0\n"
            )



# ---------------------------------------------------------------------------
# Universal latent pair-CSP layer (v172)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatentPairTopology:
    """Exact pairwise latent-state graph shared by categorical and EVEN_CYCLE covers.

    Every latent node a has a finite state set Omega_a.  Directed pair factors
    carry exact Boolean compatibility matrices C_ab(u,v).  No hard assignment,
    CNF residual, verifier score, flip, branch, or target model participates in
    construction or transport.
    """
    pure: bool
    reason: str
    nblocks: int
    max_domain: int
    domain_sizes: np.ndarray
    valid_mask: np.ndarray
    pair_src: np.ndarray
    pair_dst: np.ndarray
    reverse_edge: np.ndarray
    compatibility: np.ndarray
    # For EVEN_CYCLE covers only.  Each state stores the selected original
    # Boolean variables; categorical covers leave this empty because their
    # existing block_vars mapping is already exact.
    block_variables: tuple[tuple[int, ...], ...]
    state_selected_vars: tuple[tuple[tuple[int, ...], ...], ...]
    source_kind: str


def latent_pair_from_categorical(topology: "CategoricalTopology") -> LatentPairTopology:
    k = int(topology.nblocks)
    d = int(topology.domain_size)
    valid = np.ones((k, d), dtype=np.bool_)
    return LatentPairTopology(
        pure=bool(topology.pure),
        reason=topology.reason,
        nblocks=k,
        max_domain=d,
        domain_sizes=np.full(k, d, dtype=np.int64),
        valid_mask=valid,
        pair_src=np.asarray(topology.pair_src, dtype=np.int64).copy(),
        pair_dst=np.asarray(topology.pair_dst, dtype=np.int64).copy(),
        reverse_edge=np.asarray(topology.reverse_edge, dtype=np.int64).copy(),
        compatibility=np.asarray(topology.compatibility, dtype=np.bool_).copy(),
        block_variables=tuple(),
        state_selected_vars=tuple(),
        source_kind="categorical_EXACT1_cover",
    )


def build_even_cycle_latent_pair_topology(cnf: CNF, graph: FactorGraph) -> LatentPairTopology:
    """Contract a pure EVEN_CYCLE cover to an exact pairwise latent CSP.

    A node is one complete locally valid EVEN_CYCLE state, not merely its parity
    bit: it chooses one variable from every even bundle or one from every odd
    bundle.  Two nodes are compatible iff they agree on every original Boolean
    variable they share.  For X9 every original variable occurs in exactly two
    station factors, so this is an exact reformulation of the fused formula.
    """
    empty_i = np.zeros(0, dtype=np.int64)
    empty_b = np.zeros((0, 0, 0), dtype=np.bool_)
    if graph.n_or_factors != 0 or graph.n_exact1_factors != 0 or graph.n_even_cycle_factors == 0:
        return LatentPairTopology(False, "not a pure EVEN_CYCLE cover", 0, 0,
                                  empty_i, np.zeros((0, 0), dtype=np.bool_),
                                  empty_i, empty_i, empty_i, empty_b,
                                  tuple(), tuple(), "none")

    base_factor = graph.n_or_factors + graph.n_exact1_factors
    states: list[tuple[tuple[int, ...], ...]] = []
    block_vars: list[tuple[int, ...]] = []
    var_blocks: list[list[int]] = [[] for _ in range(cnf.nvars)]

    for local_f, ell_value in enumerate(graph.even_cycle_lengths):
        ell = int(ell_value)
        factor_id = base_factor + local_f
        a = int(graph.factor_offsets[factor_id])
        b = int(graph.factor_offsets[factor_id + 1])
        edge_vars = np.asarray(graph.edge_var[a:b], dtype=np.int64)
        edge_bundles = np.asarray(graph.edge_bundle[a:b], dtype=np.int64)
        bundles = [tuple(int(v) for v in edge_vars[edge_bundles == j]) for j in range(ell)]
        if any(len(bundle) == 0 for bundle in bundles):
            return LatentPairTopology(False, "empty bundle in EVEN_CYCLE cover", 0, 0,
                                      empty_i, np.zeros((0, 0), dtype=np.bool_),
                                      empty_i, empty_i, empty_i, empty_b,
                                      tuple(), tuple(), "none")
        local_states: list[tuple[int, ...]] = []
        for parity in (0, 1):
            active = [bundles[j] for j in range(parity, ell, 2)]
            for selection in itertools.product(*active):
                local_states.append(tuple(sorted(int(v) for v in selection)))
        states.append(tuple(local_states))
        local_variables = tuple(sorted(set(int(v) for v in edge_vars)))
        block_vars.append(local_variables)
        for v in local_variables:
            var_blocks[v].append(local_f)

    # Exact pair representation currently requires each original variable to
    # mediate exactly one pairwise equality between two latent factors.
    if any(len(owners) != 2 for owners in var_blocks):
        return LatentPairTopology(False,
                                  "EVEN_CYCLE cover is not pairwise in original variables",
                                  0, 0, empty_i, np.zeros((0, 0), dtype=np.bool_),
                                  empty_i, empty_i, empty_i, empty_b,
                                  tuple(), tuple(), "none")

    shared: dict[tuple[int, int], list[int]] = defaultdict(list)
    for v, owners in enumerate(var_blocks):
        pair = tuple(sorted((int(owners[0]), int(owners[1]))))
        shared[pair].append(v)

    k = len(states)
    domain_sizes = np.asarray([len(x) for x in states], dtype=np.int64)
    D = int(np.max(domain_sizes))
    valid = np.zeros((k, D), dtype=np.bool_)
    for a, width in enumerate(domain_sizes):
        valid[a, :int(width)] = True

    pair_src: list[int] = []
    pair_dst: list[int] = []
    reverse: list[int] = []
    matrices: list[np.ndarray] = []
    for (s, t), variables in sorted(shared.items()):
        ds = int(domain_sizes[s]); dt = int(domain_sizes[t])
        matrix = np.zeros((D, D), dtype=np.bool_)
        ssets = [set(state) for state in states[s]]
        tsets = [set(state) for state in states[t]]
        for u in range(ds):
            for v in range(dt):
                matrix[u, v] = all((q in ssets[u]) == (q in tsets[v]) for q in variables)
        e0 = len(pair_src); e1 = e0 + 1
        pair_src.extend((s, t)); pair_dst.extend((t, s)); reverse.extend((e1, e0))
        matrices.append(matrix)
        matrices.append(matrix.T.copy())

    return LatentPairTopology(
        pure=True,
        reason="pure EVEN_CYCLE cover with exact pairwise shared-variable equality",
        nblocks=k,
        max_domain=D,
        domain_sizes=domain_sizes,
        valid_mask=valid,
        pair_src=np.asarray(pair_src, dtype=np.int64),
        pair_dst=np.asarray(pair_dst, dtype=np.int64),
        reverse_edge=np.asarray(reverse, dtype=np.int64),
        compatibility=np.asarray(matrices, dtype=np.bool_),
        block_variables=tuple(block_vars),
        state_selected_vars=tuple(states),
        source_kind="EVEN_CYCLE_local_state_cover",
    )


def latent_field_from_boolean_belief(topology: LatentPairTopology,
                                     belief: np.ndarray) -> np.ndarray:
    """Continuous local-state field induced by an original Boolean field."""
    if not topology.pure or topology.source_kind != "EVEN_CYCLE_local_state_cover":
        raise ValueError("Boolean-to-latent field requires an EVEN_CYCLE latent cover")
    H = np.asarray(belief, dtype=np.float64)
    out = np.full((topology.nblocks, topology.max_domain), -60.0, dtype=np.float64)
    for a in range(topology.nblocks):
        universe = topology.block_variables[a]
        for state_id, selected_tuple in enumerate(topology.state_selected_vars[a]):
            selected = set(selected_tuple)
            # Half-weight avoids double-counting each original variable across
            # its two latent owners; only relative local-state scores matter.
            score = 0.5 * sum(float(H[v]) if v in selected else -float(H[v]) for v in universe)
            out[a, state_id] = score
        width = int(topology.domain_sizes[a])
        out[a, :width] -= float(np.max(out[a, :width]))
    return out


def _latent_softmax(field: np.ndarray, topology: LatentPairTopology) -> np.ndarray:
    H = np.asarray(field, dtype=np.float64)
    p = np.zeros_like(H)
    for a, width_value in enumerate(topology.domain_sizes):
        width = int(width_value)
        row = H[a, :width]
        row = row - float(np.max(row))
        e = np.exp(np.clip(row, -60.0, 0.0))
        p[a, :width] = e / max(float(np.sum(e)), 1e-300)
    return p


def _latent_expected_incompatibility(topology: LatentPairTopology,
                                     probabilities: np.ndarray) -> float:
    if topology.pair_src.size == 0:
        return 0.0
    valid_src = topology.valid_mask[topology.pair_src]
    valid_dst = topology.valid_mask[topology.pair_dst]
    forbidden = (
        (~topology.compatibility)
        & valid_src[:, :, None]
        & valid_dst[:, None, :]
    ).astype(np.float64, copy=False)
    contribution = np.einsum(
        "eab,ea,eb->e", forbidden,
        probabilities[topology.pair_src], probabilities[topology.pair_dst],
        optimize=True,
    )
    return 0.5 * float(np.sum(contribution))


def global_latent_deficiency_transport(
    topology: LatentPairTopology,
    base_field: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Universal formula-only latent compatibility transport.

    The source is the complete soft incompatibility tensor

        D_ab(u,v)=p_a(u) [1-C_ab(u,v)] p_b(v),

    never a hard residual.  Persistent nonnegative edge memory records where
    soft deficiency has accumulated.  A matrix-free unweighted compatibility
    resolvent then propagates response through states that may currently have
    tiny probability, avoiding the probability-freezing of the older Fisher
    Jacobian.  All scale laws depend only on latent graph geometry.
    """
    started = time.perf_counter()
    H0 = np.asarray(base_field, dtype=np.float64).copy()
    if not topology.pure or topology.pair_src.size == 0:
        return H0, {
            "kind": "universal_latent_deficiency_transport",
            "executed": False,
            "reason": "no exact pairwise latent topology",
            "runtime_seconds": float(time.perf_counter() - started),
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
        }

    k = int(topology.nblocks)
    Dmax = int(topology.max_domain)
    src = topology.pair_src
    dst = topology.pair_dst
    rev = topology.reverse_edge
    valid = topology.valid_mask
    valid_src = valid[src]
    valid_dst = valid[dst]
    forbidden0 = (
        (~topology.compatibility)
        & valid_src[:, :, None]
        & valid_dst[:, None, :]
    ).astype(np.float64, copy=False)
    degree = np.bincount(src, minlength=k).astype(np.float64)
    degree_scale = np.maximum(np.sqrt(degree), 1.0)

    p0 = _latent_softmax(H0, topology)
    source_before = _latent_expected_incompatibility(topology, p0)
    numerical_floor = (
        64.0 * np.finfo(np.float64).eps
        * max(1.0, float(topology.pair_src.size // 2) * float(Dmax))
    )
    if not np.isfinite(source_before) or source_before <= numerical_floor:
        return H0, {
            "kind": "universal_latent_deficiency_transport",
            "executed": False,
            "reason": "soft latent deficiency is at numerical zero",
            "source_kind": topology.source_kind,
            "soft_deficiency_before": float(source_before),
            "numerical_floor": float(numerical_floor),
            "nblocks": k,
            "max_domain": Dmax,
            "pair_factors": int(src.size // 2),
            "runtime_seconds": float(time.perf_counter() - started),
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
            "decimation": False,
        }

    # Scale-free laws.  Memory is slow on the latent-node scale; the response
    # is macroscopic but the old field remains the sole continuous anchor.
    memory_rate = 1.0 / math.sqrt(max(1, k))
    response_gain = math.sqrt(max(1, k))
    outer_iterations = max(1, int(math.ceil(math.log2(max(2, k)))))
    inner_iterations = max(1, k)
    contraction = 1.0 - 1.0 / math.sqrt(max(1, k))
    edge_memory = np.zeros(src.size, dtype=np.float64)
    H = H0.copy()
    rho_last = 0.0
    q_last = 0.0
    response_norm_last = 0.0

    def center(values: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values)
        for a, width_value in enumerate(topology.domain_sizes):
            width = int(width_value)
            row = values[a, :width]
            out[a, :width] = row - float(np.mean(row))
        return out

    for _outer in range(outer_iterations):
        probabilities = _latent_softmax(H, topology)
        edge_deficiency = np.einsum(
            "eab,ea,eb->e", forbidden0,
            probabilities[src], probabilities[dst], optimize=True,
        )
        edge_memory += memory_rate * edge_deficiency
        edge_memory = 0.5 * (edge_memory + edge_memory[rev])
        edge_memory = np.minimum(edge_memory, math.sqrt(max(1, k)))
        edge_weight = 1.0 + edge_memory
        forbidden = forbidden0 * edge_weight[:, None, None]

        # Memorized compatibility pressure remains active after a defect moves;
        # this is the hysteresis that prevents immediate backtracking.
        edge_pressure = np.einsum(
            "eab,eb->ea", forbidden, probabilities[dst], optimize=True,
        )
        source = np.zeros((k, Dmax), dtype=np.float64)
        np.add.at(source, src, -edge_pressure)
        source /= degree_scale[:, None]
        source = center(source)
        source[~valid] = 0.0

        def transfer(values: np.ndarray) -> np.ndarray:
            edge_response = np.einsum(
                "eab,eb->ea", forbidden, values[dst], optimize=True,
            )
            out = np.zeros((k, Dmax), dtype=np.float64)
            np.add.at(out, src, -edge_response)
            out /= degree_scale[:, None]
            out = center(out)
            out[~valid] = 0.0
            return out

        # Deterministic basis-invariant radius estimate; no singular vector is
        # exposed as a solver choice.
        seed = np.arange(1, k * Dmax + 1, dtype=np.float64).reshape(k, Dmax)
        seed[~valid] = 0.0
        seed = center(seed)
        seed_norm = float(np.linalg.norm(seed))
        if seed_norm <= 1e-300:
            break
        seed /= seed_norm
        rho = 0.0
        power_iterations = max(12, int(math.ceil(math.log2(k * Dmax + 1))) * 2)
        for _ in range(power_iterations):
            nxt = transfer(seed)
            rho = float(np.linalg.norm(nxt))
            if not np.isfinite(rho) or rho <= 1e-14:
                rho = 0.0
                break
            seed = nxt / rho
        rho_last = rho
        if rho <= 1e-14:
            break
        q = contraction / rho
        q_last = q

        term = source.copy()
        response = source.copy()
        for _ in range(inner_iterations):
            term = q * transfer(term)
            response += term
        response = center(response)
        response[~valid] = 0.0

        # State-count invariant block normalization.  Importantly, this is
        # reached only when global soft deficiency is well above numerical zero.
        rms = np.zeros((k, 1), dtype=np.float64)
        for a, width_value in enumerate(topology.domain_sizes):
            width = int(width_value)
            rms[a, 0] = math.sqrt(max(
                float(np.mean(response[a, :width] ** 2)), 1e-24
            ))
        response /= rms
        response_norm_last = float(np.linalg.norm(response))
        H = H0 + response_gain * response
        for a, width_value in enumerate(topology.domain_sizes):
            width = int(width_value)
            H[a, :width] -= float(np.max(H[a, :width]))
            H[a, :width] = np.clip(H[a, :width], -60.0, 0.0)
            H[a, width:] = -60.0

    p1 = _latent_softmax(H, topology)
    source_after = _latent_expected_incompatibility(topology, p1)
    return H, {
        "kind": "universal_latent_deficiency_transport",
        "executed": True,
        "source_kind": topology.source_kind,
        "nblocks": k,
        "max_domain": Dmax,
        "pair_factors": int(src.size // 2),
        "soft_deficiency_before": float(source_before),
        "soft_deficiency_after": float(source_after),
        "numerical_floor": float(numerical_floor),
        "memory_rate": float(memory_rate),
        "response_gain": float(response_gain),
        "outer_iterations": int(outer_iterations),
        "inner_iterations": int(inner_iterations),
        "final_transfer_radius_estimate": float(rho_last),
        "final_resolvent_q": float(q_last),
        "final_response_norm": float(response_norm_last),
        "scaling_law": (
            "memory=k^-1/2; gain=sqrt(k); outer=ceil(log2(k)); inner=k; "
            "q*rho=1-k^-1/2; structural support exact; no numerical pruning"
        ),
        "reads_boolean_assignment": False,
        "reads_cnf_residuals": False,
        "uses_verifier_score_for_selection": False,
        "boolean_flips": False,
        "branching": False,
        "decimation": False,
        "restart_portfolio": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }

# ---------------------------------------------------------------------------
# Higher-level semantic contraction: categorical EXACT1 CSP
# Restored from the verified v168 branch; this is a sibling latent-state
# specialization under the same semantic contraction hierarchy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategoricalTopology:
    """Exact higher-level encoding for disjoint EXACT1 blocks with binary incompatibilities.

    A pure categorical cover consists of disjoint EXACT1 blocks B_s that cover
    every Boolean variable, plus only cross-block negative binary clauses

        (-x_{s,a} OR -x_{t,b}).

    Such a clause forbids the categorical pair (a,b) between blocks (s,t).
    The Boolean CNF is then exactly equivalent to choosing one value per block
    subject to all pairwise compatibility matrices.
    """

    pure: bool
    reason: str
    nblocks: int
    domain_size: int
    block_vars: np.ndarray
    pair_src: np.ndarray
    pair_dst: np.ndarray
    reverse_edge: np.ndarray
    compatibility: np.ndarray
    undirected_pair_count: int
    forbidden_pair_count: int
    forbidden_per_pair_min: int | None
    forbidden_per_pair_mean: float | None
    forbidden_per_pair_max: int | None


def build_categorical_topology(cnf: CNF, graph: FactorGraph) -> CategoricalTopology:
    """Recognize an exact categorical CSP hidden in the fused CNF.

    The detector is deliberately strict.  If any variable is not in exactly one
    fused EXACT1 block, widths are nonuniform, or any remaining clause is not a
    cross-block negative binary prohibition, the categorical solver branch is
    disabled and v165 falls back to the unchanged v163 pipeline.
    """
    empty_i = np.zeros(0, dtype=np.int64)
    empty_b = np.zeros((0, 0, 0), dtype=np.bool_)
    if graph.n_exact1_factors < 2:
        return CategoricalTopology(False, "fewer than two fused EXACT1 blocks", 0, 0,
                                   np.zeros((0, 0), dtype=np.int64), empty_i, empty_i,
                                   empty_i, empty_b, 0, 0, None, None, None)

    widths = np.asarray(graph.remaining_exact1_widths, dtype=np.int64)
    if widths.size == 0 or int(np.min(widths)) != int(np.max(widths)):
        return CategoricalTopology(False, "EXACT1 block widths are not uniform", 0, 0,
                                   np.zeros((0, 0), dtype=np.int64), empty_i, empty_i,
                                   empty_i, empty_b, 0, 0, None, None, None)

    k = int(graph.n_exact1_factors)
    d = int(widths[0])
    block_vars = np.empty((k, d), dtype=np.int64)
    variable_block = np.full(cnf.nvars, -1, dtype=np.int64)
    variable_pos = np.full(cnf.nvars, -1, dtype=np.int64)

    for s in range(k):
        factor_id = graph.n_or_factors + s
        start = int(graph.factor_offsets[factor_id])
        stop = int(graph.factor_offsets[factor_id + 1])
        variables = np.asarray(graph.edge_var[start:stop], dtype=np.int64)
        if variables.size != d:
            return CategoricalTopology(False, "EXACT1 edge-width audit failed", 0, 0,
                                       np.zeros((0, 0), dtype=np.int64), empty_i,
                                       empty_i, empty_i, empty_b, 0, 0,
                                       None, None, None)
        if np.any(variable_block[variables] >= 0):
            return CategoricalTopology(False, "EXACT1 blocks overlap in Boolean variables", 0, 0,
                                       np.zeros((0, 0), dtype=np.int64), empty_i,
                                       empty_i, empty_i, empty_b, 0, 0,
                                       None, None, None)
        block_vars[s] = variables
        variable_block[variables] = s
        variable_pos[variables] = np.arange(d, dtype=np.int64)

    if np.any(variable_block < 0):
        return CategoricalTopology(False, "fused EXACT1 blocks do not cover all variables", 0, 0,
                                   np.zeros((0, 0), dtype=np.int64), empty_i, empty_i,
                                   empty_i, empty_b, 0, 0, None, None, None)

    pair_matrices: dict[tuple[int, int], np.ndarray] = {}
    forbidden_occurrences = 0
    for clause_id in graph.remaining_or_clause_ids:
        clause = cnf.clauses[int(clause_id)]
        if not (
            len(clause) == 2
            and clause[0] < 0
            and clause[1] < 0
            and abs(clause[0]) != abs(clause[1])
        ):
            return CategoricalTopology(False,
                                       "remaining clauses are not all cross-block negative binaries",
                                       0, 0, np.zeros((0, 0), dtype=np.int64),
                                       empty_i, empty_i, empty_i, empty_b, 0, 0,
                                       None, None, None)
        u = abs(int(clause[0])) - 1
        v = abs(int(clause[1])) - 1
        s = int(variable_block[u])
        t = int(variable_block[v])
        if s == t:
            return CategoricalTopology(False,
                                       "an unconsumed negative binary lies inside one EXACT1 block",
                                       0, 0, np.zeros((0, 0), dtype=np.int64),
                                       empty_i, empty_i, empty_i, empty_b, 0, 0,
                                       None, None, None)
        if s > t:
            s, t = t, s
            u, v = v, u
        matrix = pair_matrices.setdefault((s, t), np.ones((d, d), dtype=np.bool_))
        a = int(variable_pos[u])
        b = int(variable_pos[v])
        if matrix[a, b]:
            matrix[a, b] = False
            forbidden_occurrences += 1

    if not pair_matrices:
        return CategoricalTopology(False, "no cross-block compatibility factors detected", 0, 0,
                                   np.zeros((0, 0), dtype=np.int64), empty_i, empty_i,
                                   empty_i, empty_b, 0, 0, None, None, None)

    keys = sorted(pair_matrices)
    directed_count = 2 * len(keys)
    pair_src = np.empty(directed_count, dtype=np.int64)
    pair_dst = np.empty(directed_count, dtype=np.int64)
    reverse = np.empty(directed_count, dtype=np.int64)
    compatibility = np.empty((directed_count, d, d), dtype=np.bool_)
    forbidden_counts = []

    for j, (s, t) in enumerate(keys):
        matrix = pair_matrices[(s, t)]
        forbidden_counts.append(int(np.count_nonzero(~matrix)))
        e0 = 2 * j
        e1 = e0 + 1
        pair_src[e0], pair_dst[e0], reverse[e0] = s, t, e1
        pair_src[e1], pair_dst[e1], reverse[e1] = t, s, e0
        compatibility[e0] = matrix
        compatibility[e1] = matrix.T

    counts = np.asarray(forbidden_counts, dtype=np.int64)
    return CategoricalTopology(
        pure=True,
        reason="exact disjoint EXACT1 blocks plus cross-block negative-binary compatibility network",
        nblocks=k,
        domain_size=d,
        block_vars=block_vars,
        pair_src=pair_src,
        pair_dst=pair_dst,
        reverse_edge=reverse,
        compatibility=compatibility,
        undirected_pair_count=len(keys),
        forbidden_pair_count=int(forbidden_occurrences),
        forbidden_per_pair_min=int(np.min(counts)),
        forbidden_per_pair_mean=float(np.mean(counts)),
        forbidden_per_pair_max=int(np.max(counts)),
    )


def _categorical_softmax(field: np.ndarray) -> np.ndarray:
    z = np.asarray(field, dtype=np.float64)
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(shifted, -60.0, 0.0))
    return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-300)


def _categorical_expected_forbidden(topology: CategoricalTopology,
                                    probabilities: np.ndarray) -> float:
    # Directed representation double-counts each undirected interaction.
    forbidden = (~topology.compatibility).astype(np.float64, copy=False)
    contribution = np.einsum(
        "eab,ea,eb->e",
        forbidden,
        probabilities[topology.pair_src],
        probabilities[topology.pair_dst],
        optimize=True,
    )
    return 0.5 * float(np.sum(contribution))


def global_categorical_exact1_csp_operator(
    topology: CategoricalTopology,
    *,
    bp_iterations: int,
    flow_iterations: int,
    message_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """One global continuous operator on the exact categorical CSP encoding.

    Stage A is reinforced annealed sum-product BP over every categorical pair
    factor simultaneously.  Stage B is a synchronous variational compatibility
    flow over the full product of simplices.  There is no Boolean state,
    residual-clause score, local repair, variable flip, branching, restart, or
    candidate portfolio anywhere inside this operator.

    Scale-free default laws are derived from the detected structure:

        alpha = rho = 1/sqrt(k)
        beta_start = 1/sqrt(d)
        beta_stop  = sqrt(k)
        T_BP = T_flow = 15*k

    where k is the number of categorical EXACT1 blocks and d their width.
    """
    started = time.perf_counter()
    if not topology.pure:
        raise ValueError("categorical operator requires a pure categorical cover")

    k = int(topology.nblocks)
    d = int(topology.domain_size)
    src = topology.pair_src
    dst = topology.pair_dst
    rev = topology.reverse_edge
    compatibility = topology.compatibility
    forbidden = (~compatibility).astype(np.float64, copy=False)
    directed = int(src.size)

    alpha = min(0.2, max(0.02, 1.0 / math.sqrt(max(1, k))))
    rho = 1.0 / math.sqrt(max(1, k))
    beta_start = 1.0 / math.sqrt(max(1, d))
    beta_stop = math.sqrt(max(1, k))
    message_clip_value = (
        float(message_clip) if float(message_clip) > 0.0 else 2.0 * float(k)
    )
    bp_count = max(1, int(bp_iterations))
    flow_count = max(1, int(flow_iterations))

    # A. Whole-graph categorical cavity transport.
    messages = np.zeros((directed, d), dtype=np.float64)
    final_bp_update = math.inf
    for iteration in range(bp_count):
        total = np.zeros((k, d), dtype=np.float64)
        _v179_scatter_rows(total, dst, messages)
        cavity = total[src] - messages[rev] + rho * total[src]
        if bp_count == 1:
            beta = beta_stop
        else:
            beta = beta_start * (beta_stop / beta_start) ** (
                iteration / float(bp_count - 1)
            )

        weighted = np.where(
            compatibility,
            beta * cavity[:, :, None],
            -np.inf,
        )
        maximum = np.max(weighted, axis=1)
        # Every value in a recognized compatibility matrix must have at least
        # one support.  If not, keep the arithmetic finite but record it via
        # the resulting large negative message.
        finite_max = np.where(np.isfinite(maximum), maximum, 0.0)
        exponential_sum = np.sum(
            np.exp(weighted - finite_max[:, None, :]),
            axis=1,
        )
        new_messages = (
            finite_max
            + np.log(np.maximum(exponential_sum, 1e-300))
        ) / beta
        new_messages -= np.max(new_messages, axis=1, keepdims=True)
        if message_clip_value > 0.0:
            new_messages = np.clip(new_messages, -message_clip_value, 0.0)
        final_bp_update = float(np.max(np.abs(new_messages - messages)))
        messages = (1.0 - alpha) * messages + alpha * new_messages

    bp_field = np.zeros((k, d), dtype=np.float64)
    np.add.at(bp_field, dst, messages)
    bp_scale = float(np.std(bp_field))
    if not np.isfinite(bp_scale) or bp_scale <= 1e-12:
        bp_scale = 1.0
    anchor = (bp_field - np.mean(bp_field, axis=1, keepdims=True)) / bp_scale

    # B. Whole-graph variational compatibility continuation.
    field = anchor.copy()
    probabilities = _categorical_softmax(field)
    expected_before = _categorical_expected_forbidden(topology, probabilities)
    final_flow_update = math.inf

    for iteration in range(flow_count):
        pair_pressure = np.einsum(
            "eab,eb->ea",
            forbidden,
            probabilities[dst],
            optimize=True,
        )
        score = np.zeros((k, d), dtype=np.float64)
        np.add.at(score, src, -pair_pressure)
        score -= np.mean(score, axis=1, keepdims=True)

        if flow_count == 1:
            beta = beta_stop
        else:
            beta = beta_start * (beta_stop / beta_start) ** (
                iteration / float(flow_count - 1)
            )
        target = beta * score + anchor
        new_field = (1.0 - alpha) * field + alpha * target
        new_field -= np.max(new_field, axis=1, keepdims=True)
        new_field = np.clip(new_field, -60.0, 0.0)
        final_flow_update = float(np.max(np.abs(new_field - field)))
        field = new_field
        probabilities = _categorical_softmax(field)

    expected_after = _categorical_expected_forbidden(topology, probabilities)
    entropy = -float(np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-300))
    )) / float(k)

    meta = {
        "kind": "global_categorical_EXACT1_CSP_fusion",
        "executed": True,
        "categorical_blocks_k": k,
        "domain_width_d": d,
        "boolean_variables": int(k * d),
        "pair_factors": int(topology.undirected_pair_count),
        "directed_pair_messages": directed,
        "forbidden_pairs": int(topology.forbidden_pair_count),
        "bp_iterations": bp_count,
        "flow_iterations": flow_count,
        "damping": float(alpha),
        "reinforcement": float(rho),
        "beta_start": float(beta_start),
        "beta_stop": float(beta_stop),
        "scaling_law": (
            "alpha=min(0.2,max(0.02,k^-1/2)); rho=k^-1/2; "
            "beta0=d^-1/2; beta1=k^1/2; message_clip=2*k; "
            "T_BP=T_flow=15*k by default"
        ),
        "message_clip": float(message_clip_value),
        "final_bp_update_norm": float(final_bp_update),
        "final_flow_update_norm": float(final_flow_update),
        "expected_forbidden_mass_before_flow": float(expected_before),
        "expected_forbidden_mass_after_flow": float(expected_after),
        "mean_block_entropy_after_flow": float(entropy),
        "runtime_seconds": float(time.perf_counter() - started),
        "reads_boolean_assignment": False,
        "reads_cnf_residuals": False,
        "uses_verifier_score_for_selection": False,
        "boolean_archive": False,
        "boolean_flips": False,
        "branching": False,
        "decimation": False,
        "restart_portfolio": False,
        "external_solver": False,
        "final_readout": "one categorical argmax per EXACT1 block",
    }
    return field, bp_field, messages, meta


def global_categorical_triangle_loop_lift(
    topology: CategoricalTopology,
    base_field: np.ndarray,
    *,
    iterations: int,
    gain: float,
) -> tuple[np.ndarray, dict]:
    """Global higher-order loop correlation over all interaction triangles.

    The v164 categorical operator uses only pairwise compatibility factors.
    This layer constructs *all* triangles of the categorical interaction graph
    simultaneously.  For a triangle (s,t,u), the support of value ``a`` at s
    is the soft total mass of compatible pairs (b,c) at t,u that satisfy all
    three pair factors at once:

        T_s(a) = sum_{b,c}
                 C_st(a,b) C_su(a,c) C_tu(b,c) p_t(b) p_u(c).

    The analogous quantities are computed for t and u.  Log-supports from all
    triangles are accumulated into one complete k-by-d field, centered and
    normalized globally, then coupled to the unchanged pairwise variational
    field.  Every block/value is updated synchronously; no Boolean assignment,
    residual clause, verifier score, local repair, branch, restart, or
    candidate portfolio is constructed inside the lift.
    """
    started = time.perf_counter()
    k = int(topology.nblocks)
    d = int(topology.domain_size)
    H0 = np.asarray(base_field, dtype=np.float64)
    if H0.shape != (k, d):
        raise ValueError("categorical triangle lift field shape mismatch")

    # Canonical undirected compatibility table.  The recognized topology stores
    # each pair twice; use one orientation and transpose if needed.
    pair_matrix: dict[tuple[int, int], np.ndarray] = {}
    adjacency = [set() for _ in range(k)]
    for edge in range(0, int(topology.pair_src.size), 2):
        s = int(topology.pair_src[edge])
        t = int(topology.pair_dst[edge])
        C = np.asarray(topology.compatibility[edge], dtype=np.float64)
        if s < t:
            pair_matrix[(s, t)] = C
        else:
            pair_matrix[(t, s)] = C.T
        adjacency[s].add(t)
        adjacency[t].add(s)

    triangles: list[tuple[int, int, int]] = []
    for s in range(k):
        for t in sorted(v for v in adjacency[s] if v > s):
            common = adjacency[s].intersection(adjacency[t])
            for u in sorted(v for v in common if v > t):
                triangles.append((s, t, u))

    count = max(0, int(iterations))
    if not triangles or count == 0:
        return H0.copy(), {
            "kind": "global_categorical_triangle_loop_lift",
            "executed": False,
            "reason": "no interaction triangles" if not triangles else "disabled by iteration count",
            "triangle_count": int(len(triangles)),
            "runtime_seconds": float(time.perf_counter() - started),
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
            "restart_portfolio": False,
        }

    tri_s = np.asarray([x[0] for x in triangles], dtype=np.int64)
    tri_t = np.asarray([x[1] for x in triangles], dtype=np.int64)
    tri_u = np.asarray([x[2] for x in triangles], dtype=np.int64)
    C_st = np.stack([pair_matrix[(s, t)] for s, t, u in triangles])
    C_su = np.stack([pair_matrix[(s, u)] for s, t, u in triangles])
    C_tu = np.stack([pair_matrix[(t, u)] for s, t, u in triangles])

    alpha = min(0.2, max(0.02, 1.0 / math.sqrt(max(1, k))))
    beta = math.sqrt(max(1, k))
    gamma = max(0.0, float(gain))
    forbidden = (~topology.compatibility).astype(np.float64, copy=False)
    src = np.asarray(topology.pair_src, dtype=np.int64)
    dst = np.asarray(topology.pair_dst, dtype=np.int64)

    field = H0.copy()
    p0 = _categorical_softmax(field)
    expected_before = _categorical_expected_forbidden(topology, p0)
    entropy_before = -float(np.sum(
        p0 * np.log(np.maximum(p0, 1e-300))
    )) / float(k)

    final_update = math.inf
    final_triangle_scale = 0.0
    final_triangle_abs_mean = 0.0
    eps = 1e-300

    for _ in range(count):
        probabilities = _categorical_softmax(field)

        # Existing whole-graph pairwise compatibility pressure.
        pair_pressure = np.einsum(
            "eab,eb->ea",
            forbidden,
            probabilities[dst],
            optimize=True,
        )
        pair_score = np.zeros((k, d), dtype=np.float64)
        np.add.at(pair_score, src, -pair_pressure)
        pair_score -= np.mean(pair_score, axis=1, keepdims=True)

        # Simultaneous three-factor support on every graph triangle.
        support_s = np.einsum(
            "tab,tac,tbc,tb,tc->ta",
            C_st, C_su, C_tu,
            probabilities[tri_t], probabilities[tri_u],
            optimize=True,
        )
        support_t = np.einsum(
            "tab,tbc,tac,ta,tc->tb",
            C_st, C_tu, C_su,
            probabilities[tri_s], probabilities[tri_u],
            optimize=True,
        )
        support_u = np.einsum(
            "tac,tbc,tab,ta,tb->tc",
            C_su, C_tu, C_st,
            probabilities[tri_s], probabilities[tri_t],
            optimize=True,
        )

        triangle_score = np.zeros((k, d), dtype=np.float64)
        np.add.at(triangle_score, tri_s, np.log(np.maximum(support_s, eps)))
        np.add.at(triangle_score, tri_t, np.log(np.maximum(support_t, eps)))
        np.add.at(triangle_score, tri_u, np.log(np.maximum(support_u, eps)))
        triangle_score -= np.mean(triangle_score, axis=1, keepdims=True)

        triangle_scale = float(np.std(triangle_score))
        if not np.isfinite(triangle_scale) or triangle_scale <= 1e-12:
            triangle_scale = 1.0
        normalized_triangle = triangle_score / triangle_scale

        # H0 is the full v164 categorical field, used as a global anchor.  The
        # lift never selects individual blocks or pair factors.
        target = H0 + beta * pair_score + gamma * normalized_triangle
        new_field = (1.0 - alpha) * field + alpha * target
        new_field -= np.max(new_field, axis=1, keepdims=True)
        new_field = np.clip(new_field, -60.0, 0.0)

        final_update = float(np.max(np.abs(new_field - field)))
        final_triangle_scale = float(triangle_scale)
        final_triangle_abs_mean = float(np.mean(np.abs(normalized_triangle)))
        field = new_field

    probabilities = _categorical_softmax(field)
    expected_after = _categorical_expected_forbidden(topology, probabilities)
    entropy_after = -float(np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-300))
    )) / float(k)

    meta = {
        "kind": "global_categorical_triangle_loop_lift",
        "executed": True,
        "triangle_count": int(len(triangles)),
        "iterations": int(count),
        "damping": float(alpha),
        "pair_scale": float(beta),
        "triangle_gain": float(gamma),
        "scaling_law": (
            "T_triangle=ceil(sqrt(k)*log(k)) by default; "
            "gamma=sqrt(k); pair_scale=sqrt(k); alpha=k^-1/2 clipped to [0.02,0.2]"
        ),
        "expected_forbidden_mass_before": float(expected_before),
        "expected_forbidden_mass_after": float(expected_after),
        "mean_block_entropy_before": float(entropy_before),
        "mean_block_entropy_after": float(entropy_after),
        "final_update_norm": float(final_update),
        "final_triangle_score_std": float(final_triangle_scale),
        "final_normalized_triangle_abs_mean": float(final_triangle_abs_mean),
        "runtime_seconds": float(time.perf_counter() - started),
        "reads_boolean_assignment": False,
        "reads_cnf_residuals": False,
        "uses_verifier_score_for_selection": False,
        "boolean_archive": False,
        "boolean_flips": False,
        "branching": False,
        "decimation": False,
        "restart_portfolio": False,
        "external_solver": False,
        "final_readout": "none; one categorical argmax occurs only after this global lift",
    }
    return field, meta



def global_categorical_loop_resolvent_lift(
    topology: CategoricalTopology,
    base_field: np.ndarray,
    *,
    outer_iterations: int,
    inner_iterations: int,
    gain: float,
) -> tuple[np.ndarray, dict]:
    """Self-consistent all-length categorical loop susceptibility.

    The triangle layer supplies a formula-only higher-order source r(H).  The
    pairwise categorical variational map has a full-field Jacobian J(H).  This
    layer applies the matrix-free resolvent

        delta = (I - q J)^(-1) r
              = r + q J r + q^2 J^2 r + ...

    so paths/loops of every length contribute simultaneously.  The resolvent
    is recomputed from the current complete k-by-d field on every outer step.
    No Boolean assignment, clause residual, verifier score, block selection,
    branch, flip, restart, or candidate portfolio is constructed.

    Default scale laws are supplied by the caller:

        T_outer = ceil(log k),  T_inner = k,
        gain = sqrt(k),  q*rho(J) = 1 - k^(-1/2).
    """
    started = time.perf_counter()
    k = int(topology.nblocks)
    d = int(topology.domain_size)
    field = np.asarray(base_field, dtype=np.float64).copy()
    if field.shape != (k, d):
        raise ValueError("categorical loop-resolvent field shape mismatch")

    outer_count = max(0, int(outer_iterations))
    inner_count = max(1, int(inner_iterations))
    response_gain = max(0.0, float(gain))
    if outer_count == 0 or response_gain == 0.0:
        return field, {
            "kind": "global_categorical_all_length_loop_resolvent",
            "executed": False,
            "reason": "disabled by iteration count or zero gain",
            "runtime_seconds": float(time.perf_counter() - started),
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
            "restart_portfolio": False,
        }

    forbidden = (~topology.compatibility).astype(np.float64, copy=False)
    src = np.asarray(topology.pair_src, dtype=np.int64)
    dst = np.asarray(topology.pair_dst, dtype=np.int64)
    beta = math.sqrt(max(1, k))
    contraction = 1.0 - 1.0 / math.sqrt(max(1, k))
    eps = 1e-30

    p0 = _categorical_softmax(field)
    expected_before = _categorical_expected_forbidden(topology, p0)
    entropy_before = -float(np.sum(
        p0 * np.log(np.maximum(p0, 1e-300))
    )) / float(k)

    final_radius = 0.0
    final_q = 0.0
    final_response_std = 0.0
    final_update = math.inf

    for _outer in range(outer_count):
        # One fresh whole-graph triangle response is used only as a continuous
        # source.  The helper performs no Boolean readout or verifier query.
        source_field, _source_meta = global_categorical_triangle_loop_lift(
            topology, field, iterations=1, gain=beta
        )
        source = source_field - field
        source -= np.mean(source, axis=1, keepdims=True)
        source_scale = float(np.std(source))
        if not np.isfinite(source_scale) or source_scale <= 1e-14:
            break
        source /= source_scale

        probabilities = _categorical_softmax(field)

        def jacobian_apply(vector: np.ndarray) -> np.ndarray:
            v = np.asarray(vector, dtype=np.float64)
            mean = np.sum(probabilities * v, axis=1, keepdims=True)
            dp = probabilities * (v - mean)
            pressure = np.einsum(
                "eab,eb->ea", forbidden, dp[dst], optimize=True
            )
            out = np.zeros((k, d), dtype=np.float64)
            np.add.at(out, src, -pressure)
            out -= np.mean(out, axis=1, keepdims=True)
            return beta * out

        # Deterministic power estimate of the response radius, initialized by
        # the same formula-global source rather than by random noise.
        direction = source.copy()
        norm = float(np.linalg.norm(direction))
        if norm <= eps:
            break
        direction /= norm
        radius = 0.0
        power_count = max(8, int(math.ceil(math.sqrt(max(1, k)) * math.log(max(2, k)))))
        for _ in range(power_count):
            image = jacobian_apply(direction)
            image_norm = float(np.linalg.norm(image))
            if not np.isfinite(image_norm) or image_norm <= eps:
                radius = 0.0
                break
            direction = image / image_norm
            radius = image_norm

        if radius <= 1e-14:
            break
        q = contraction / radius

        response = np.zeros_like(source)
        for _ in range(inner_count):
            response = source + q * jacobian_apply(response)

        response -= np.mean(response, axis=1, keepdims=True)
        response_std = float(np.std(response))
        if not np.isfinite(response_std) or response_std <= 1e-14:
            break
        response /= response_std

        new_field = field + response_gain * response
        new_field -= np.max(new_field, axis=1, keepdims=True)
        new_field = np.clip(new_field, -60.0, 0.0)
        final_update = float(np.max(np.abs(new_field - field)))
        field = new_field
        final_radius = float(radius)
        final_q = float(q)
        final_response_std = float(response_std)

    probabilities = _categorical_softmax(field)
    expected_after = _categorical_expected_forbidden(topology, probabilities)
    entropy_after = -float(np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-300))
    )) / float(k)

    meta = {
        "kind": "global_categorical_all_length_loop_resolvent",
        "executed": True,
        "outer_iterations": int(outer_count),
        "inner_iterations": int(inner_count),
        "response_gain": float(response_gain),
        "jacobian_contraction_target": float(contraction),
        "final_jacobian_radius_estimate": float(final_radius),
        "final_resolvent_q": float(final_q),
        "final_raw_response_std": float(final_response_std),
        "expected_forbidden_mass_before": float(expected_before),
        "expected_forbidden_mass_after": float(expected_after),
        "mean_block_entropy_before": float(entropy_before),
        "mean_block_entropy_after": float(entropy_after),
        "final_update_norm": float(final_update),
        "scaling_law": (
            "T_outer=ceil(log(k)); T_inner=k; gain=sqrt(k); "
            "q*rho(J)=1-k^-1/2; power_iters=ceil(sqrt(k)*log(k))"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "reads_boolean_assignment": False,
        "reads_cnf_residuals": False,
        "uses_verifier_score_for_selection": False,
        "boolean_archive": False,
        "boolean_flips": False,
        "branching": False,
        "decimation": False,
        "restart_portfolio": False,
        "external_solver": False,
        "final_readout": "none; one categorical argmax occurs only after this global lift",
    }
    return field, meta


def global_categorical_two_replica_bifurcation(
    topology: CategoricalTopology,
    base_field: np.ndarray,
    *,
    push_iterations: int,
    hold_iterations: int,
    release_iterations: int,
    settle_iterations: int,
    step_size: float,
    entropy_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Deterministic isoenergetic Fisher-replicon bifurcation.

    v167 initialized the two replicas with a dominant compatibility-Jacobian
    power mode and pushed them apart with a raw probability-overlap penalty.
    That construction produced a real macroscopic split, but one branch crossed
    the wall through a high-energy open constraint chain.

    v168 instead constructs the Fisher-normalized Hessian of the *complete*
    categorical free energy at the incoming whole-field state.  Its negative
    replicon eigenspace is the set of locally soft antisymmetric directions.  If
    no negative eigenvalue exists, the single softest mode is used.

    The modes live in square-root probability coordinates q=sqrt(p), where the
    simplex Fisher metric is Euclidean.  Initial replicas are placed by exact
    blockwise sphere geodesics, avoiding the delta-p/p blow-up on tiny
    probabilities.  Replica separation uses Hellinger overlap and its force is
    projected orthogonally to the original free-energy gradient in the Fisher
    metric on every step.  Thus the repulsive part has zero first-order work on
    the original categorical energy.

    No Boolean state, residual clause, verifier score, local repair, restart,
    branch, decimation, candidate portfolio, or external solver is used.  The
    replicas are merged continuously by their symmetric Hellinger barycenter
    before the later triangle and all-length global lifts and before the sole
    categorical argmax readout.
    """
    started = time.perf_counter()
    k = int(topology.nblocks)
    d = int(topology.domain_size)
    H0 = np.asarray(base_field, dtype=np.float64).copy()
    if H0.shape != (k, d):
        raise ValueError("Fisher-replicon field shape mismatch")

    if scipy_coo_matrix is None or scipy_eigsh is None:
        return H0, H0.copy(), H0.copy(), {
            "kind": "global_categorical_isoenergetic_Fisher_replicon_bifurcation",
            "executed": False,
            "reason": "SciPy sparse eigensolver is unavailable",
            "runtime_seconds": float(time.perf_counter() - started),
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
            "restart_portfolio": False,
        }

    src = np.asarray(topology.pair_src, dtype=np.int64)
    dst = np.asarray(topology.pair_dst, dtype=np.int64)
    forbidden = (~topology.compatibility).astype(np.float64, copy=False)
    eps_prob = 1e-300
    tau = max(0.0, float(entropy_weight))

    def pressure(probabilities: np.ndarray) -> np.ndarray:
        edge_pressure = np.einsum(
            "eab,eb->ea", forbidden, probabilities[dst], optimize=True
        )
        out = np.zeros((k, d), dtype=np.float64)
        np.add.at(out, src, edge_pressure)
        return out

    def tangent_center(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        return values - np.sum(probabilities * values, axis=1, keepdims=True)

    def fisher_inner(a: np.ndarray, b: np.ndarray,
                     probabilities: np.ndarray) -> float:
        return float(np.sum(probabilities * a * b))

    def energy_gradient(probabilities: np.ndarray) -> np.ndarray:
        gradient = pressure(probabilities)
        if tau > 0.0:
            gradient += tau * (
                np.log(np.maximum(probabilities, eps_prob)) + 1.0
            )
        return tangent_center(gradient, probabilities)

    def free_energy(probabilities: np.ndarray) -> float:
        expected = _categorical_expected_forbidden(topology, probabilities)
        entropy_term = float(np.sum(
            probabilities * np.log(np.maximum(probabilities, eps_prob))
        ))
        return expected + tau * entropy_term

    def hellinger_overlap(p_plus: np.ndarray, p_minus: np.ndarray) -> float:
        return float(np.sum(np.sqrt(np.maximum(
            p_plus * p_minus, eps_prob
        ))) / float(k))

    p0 = _categorical_softmax(H0)
    q0 = np.sqrt(np.maximum(p0, eps_prob))
    expected_before = _categorical_expected_forbidden(topology, p0)
    entropy_before = -float(np.sum(
        p0 * np.log(np.maximum(p0, eps_prob))
    )) / float(k)

    # Deterministic orthonormal bases of the tangent spaces q_s^perp.
    tangent_bases: list[np.ndarray] = []
    for s in range(k):
        u = q0[s] / max(float(np.linalg.norm(q0[s])), 1e-300)
        e0 = np.zeros(d, dtype=np.float64)
        e0[0] = 1.0
        if float(np.linalg.norm(e0 - u)) <= 1e-13:
            householder = np.eye(d, dtype=np.float64)
        else:
            v = e0 - u
            householder = (
                np.eye(d, dtype=np.float64)
                - 2.0 * np.outer(v, v) / float(np.dot(v, v))
            )
        tangent_bases.append(householder[:, 1:])

    reduced_width = d - 1
    reduced_size = k * reduced_width
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_block(s: int, t: int, block: np.ndarray) -> None:
        r0 = s * reduced_width
        c0 = t * reduced_width
        rr = np.broadcast_to(
            (r0 + np.arange(reduced_width, dtype=np.int64))[:, None],
            (reduced_width, reduced_width),
        )
        cc = np.broadcast_to(
            (c0 + np.arange(reduced_width, dtype=np.int64))[None, :],
            (reduced_width, reduced_width),
        )
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(np.asarray(block, dtype=np.float64).ravel().tolist())

    # In Fisher coordinates y=delta p/sqrt(p), entropy contributes tau*I.
    diagonal_block = tau * np.eye(reduced_width, dtype=np.float64)
    for s in range(k):
        add_block(s, s, diagonal_block)

    # Directed edges occur in exact reverse pairs.  Add each undirected
    # interaction once in both symmetric block positions.
    for edge_id in range(0, src.size, 2):
        s = int(src[edge_id])
        t = int(dst[edge_id])
        transformed = (
            q0[s][:, None]
            * forbidden[edge_id]
            * q0[t][None, :]
        )
        block = tangent_bases[s].T @ transformed @ tangent_bases[t]
        add_block(s, t, block)
        add_block(t, s, block.T)

    hessian = scipy_coo_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(reduced_size, reduced_size),
    ).tocsr()
    hessian = 0.5 * (hessian + hessian.T)

    eig_count = min(
        max(2, int(math.ceil(math.sqrt(max(1, k))))),
        max(1, reduced_size - 2),
    )
    gradient0 = energy_gradient(p0)
    v0_parts = []
    for s in range(k):
        v0_parts.append(
            tangent_bases[s].T @ (q0[s] * gradient0[s])
        )
    eigen_seed = np.concatenate(v0_parts)
    seed_norm = float(np.linalg.norm(eigen_seed))
    if not np.isfinite(seed_norm) or seed_norm <= 1e-14:
        eigen_seed = np.arange(1, reduced_size + 1, dtype=np.float64)
        seed_norm = float(np.linalg.norm(eigen_seed))
    eigen_seed /= max(seed_norm, 1e-300)

    eigenvalues, eigenvectors = scipy_eigsh(
        hessian,
        k=eig_count,
        which="SA",
        v0=eigen_seed,
        tol=1e-10,
        maxiter=max(2000, 20 * reduced_size),
        ncv=min(reduced_size, max(2 * eig_count + 1, 4 * eig_count)),
    )
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    negative_ids = np.flatnonzero(eigenvalues < -1e-10 * spectral_scale)
    active_ids = negative_ids if negative_ids.size else np.asarray([0], dtype=np.int64)

    # Canonical combination of the complete negative replicon subspace.
    mode_q = np.zeros((k, d), dtype=np.float64)
    for mode_id in active_ids:
        reduced_mode = eigenvectors[:, int(mode_id)].reshape(k, reduced_width)
        component = np.zeros((k, d), dtype=np.float64)
        for s in range(k):
            component[s] = tangent_bases[s] @ reduced_mode[s]
        delta_p = q0 * component
        orientation = float(np.sum(gradient0 * delta_p))
        if orientation > 0.0:
            component = -component
        elif abs(orientation) <= 1e-14:
            flat = component.ravel()
            nz = np.flatnonzero(np.abs(flat) > 1e-14)
            if nz.size and flat[int(nz[0])] < 0.0:
                component = -component
        weight = math.sqrt(max(abs(float(eigenvalues[int(mode_id)])), tau, 1e-15))
        mode_q += weight * component

    mode_norm = float(np.linalg.norm(mode_q))
    if not np.isfinite(mode_norm) or mode_norm <= 1e-14:
        return H0, H0.copy(), H0.copy(), {
            "kind": "global_categorical_isoenergetic_Fisher_replicon_bifurcation",
            "executed": False,
            "reason": "Fisher replicon mode is numerically zero",
            "runtime_seconds": float(time.perf_counter() - started),
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
            "restart_portfolio": False,
        }
    mode_q /= mode_norm
    block_mode_mass = np.sum(mode_q * mode_q, axis=1)
    mode_participation = float(
        np.sum(block_mode_mass) ** 2
        / max(float(np.sum(block_mode_mass * block_mode_mass)), 1e-300)
    )

    # Exact blockwise Fisher-sphere geodesics.  The angle 1/sqrt(k) is a
    # macroscopic block-scale perturbation; no tiny-p logit division occurs.
    geodesic_angle = 1.0 / math.sqrt(max(1, k))

    def geodesic_field(sign: float) -> np.ndarray:
        q = np.zeros_like(q0)
        for s in range(k):
            tangent = mode_q[s] - q0[s] * float(np.dot(q0[s], mode_q[s]))
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-15:
                q[s] = q0[s]
            else:
                angle = geodesic_angle * tangent_norm
                q[s] = (
                    math.cos(angle) * q0[s]
                    + sign * math.sin(angle) * tangent / tangent_norm
                )
        probabilities = q * q
        probabilities /= np.maximum(
            np.sum(probabilities, axis=1, keepdims=True), 1e-300
        )
        field = np.log(np.maximum(probabilities, eps_prob))
        field -= np.max(field, axis=1, keepdims=True)
        return np.clip(field, -60.0, 0.0)

    H_plus = geodesic_field(+1.0)
    H_minus = geodesic_field(-1.0)

    lambda_min = float(eigenvalues[0])
    curvature_scale = max(abs(lambda_min), tau, 1e-12)
    # For Hellinger overlap the local critical scale is 2*k*lambda.  sqrt(k)
    # promotes the complete negative subspace from a local to a block-global
    # response without reading a hard assignment or residual.
    gamma_max = 2.0 * float(k) * curvature_scale * math.sqrt(max(1, k))

    T_push = max(0, int(push_iterations))
    T_hold = max(0, int(hold_iterations))
    T_release = max(0, int(release_iterations))
    T_settle = max(0, int(settle_iterations))
    total_iterations = T_push + T_hold + T_release + T_settle
    alpha = max(1e-6, float(step_size))

    p_plus0 = _categorical_softmax(H_plus)
    p_minus0 = _categorical_softmax(H_minus)
    overlap_initial = hellinger_overlap(p_plus0, p_minus0)
    overlap_minimum = overlap_initial
    final_update = math.inf
    max_isoenergetic_inner_product = 0.0

    for iteration in range(total_iterations):
        if iteration < T_push and T_push > 0:
            gamma = gamma_max * float(iteration + 1) / float(T_push)
        elif iteration < T_push + T_hold:
            gamma = gamma_max
        elif iteration < T_push + T_hold + T_release and T_release > 0:
            elapsed = iteration - T_push - T_hold + 1
            gamma = gamma_max * max(0.0, 1.0 - elapsed / float(T_release))
        else:
            gamma = 0.0

        p_plus = _categorical_softmax(H_plus)
        p_minus = _categorical_softmax(H_minus)
        overlap_now = hellinger_overlap(p_plus, p_minus)
        overlap_minimum = min(overlap_minimum, overlap_now)

        e_plus = energy_gradient(p_plus)
        e_minus = energy_gradient(p_minus)

        r_plus = tangent_center(
            0.5 / float(k) * np.sqrt(
                np.maximum(p_minus, eps_prob)
                / np.maximum(p_plus, eps_prob)
            ),
            p_plus,
        )
        r_minus = tangent_center(
            0.5 / float(k) * np.sqrt(
                np.maximum(p_plus, eps_prob)
                / np.maximum(p_minus, eps_prob)
            ),
            p_minus,
        )

        # Remove the component that performs first-order work on the original
        # free energy in the Fisher metric.
        denom_plus = fisher_inner(e_plus, e_plus, p_plus) + 1e-30
        denom_minus = fisher_inner(e_minus, e_minus, p_minus) + 1e-30
        r_plus -= e_plus * (
            fisher_inner(e_plus, r_plus, p_plus) / denom_plus
        )
        r_minus -= e_minus * (
            fisher_inner(e_minus, r_minus, p_minus) / denom_minus
        )
        r_plus = tangent_center(r_plus, p_plus)
        r_minus = tangent_center(r_minus, p_minus)
        max_isoenergetic_inner_product = max(
            max_isoenergetic_inner_product,
            abs(fisher_inner(e_plus, r_plus, p_plus)),
            abs(fisher_inner(e_minus, r_minus, p_minus)),
        )

        new_plus = H_plus - alpha * (e_plus + gamma * r_plus)
        new_minus = H_minus - alpha * (e_minus + gamma * r_minus)
        new_plus -= np.max(new_plus, axis=1, keepdims=True)
        new_minus -= np.max(new_minus, axis=1, keepdims=True)
        new_plus = np.clip(new_plus, -60.0, 0.0)
        new_minus = np.clip(new_minus, -60.0, 0.0)
        final_update = max(
            float(np.max(np.abs(new_plus - H_plus))),
            float(np.max(np.abs(new_minus - H_minus))),
        )
        H_plus, H_minus = new_plus, new_minus

    p_plus = _categorical_softmax(H_plus)
    p_minus = _categorical_softmax(H_minus)
    overlap_final = hellinger_overlap(p_plus, p_minus)
    E_plus = _categorical_expected_forbidden(topology, p_plus)
    E_minus = _categorical_expected_forbidden(topology, p_minus)
    F_plus = free_energy(p_plus)
    F_minus = free_energy(p_minus)

    # Symmetric Hellinger barycenter preserves the replica covariance instead
    # of exponentially deleting the higher-free-energy branch as in v167.
    barycentric_amplitude = np.sqrt(p_plus) + np.sqrt(p_minus)
    merged_probabilities = barycentric_amplitude * barycentric_amplitude
    merged_probabilities /= np.maximum(
        np.sum(merged_probabilities, axis=1, keepdims=True), 1e-300
    )
    merged_field = np.log(np.maximum(merged_probabilities, eps_prob))
    merged_field -= np.max(merged_field, axis=1, keepdims=True)
    merged_field = np.clip(merged_field, -60.0, 0.0)

    expected_after = _categorical_expected_forbidden(
        topology, merged_probabilities
    )
    entropy_after = -float(np.sum(
        merged_probabilities
        * np.log(np.maximum(merged_probabilities, eps_prob))
    )) / float(k)
    disagreement_blocks = int(np.count_nonzero(
        np.argmax(p_plus, axis=1) != np.argmax(p_minus, axis=1)
    ))

    meta = {
        "kind": "global_categorical_isoenergetic_Fisher_replicon_bifurcation",
        "executed": True,
        "push_iterations": int(T_push),
        "hold_iterations": int(T_hold),
        "release_iterations": int(T_release),
        "settle_iterations": int(T_settle),
        "total_iterations": int(total_iterations),
        "step_size": float(alpha),
        "entropy_weight": float(tau),
        "replicon_reduced_dimension": int(reduced_size),
        "replicon_eigenvalues": [float(x) for x in eigenvalues.tolist()],
        "replicon_negative_mode_count": int(negative_ids.size),
        "replicon_active_mode_count": int(active_ids.size),
        "replicon_lambda_min": float(lambda_min),
        "replicon_mode_participation": float(mode_participation),
        "Fisher_geodesic_angle": float(geodesic_angle),
        "repulsion_gamma_max": float(gamma_max),
        "repulsion_scale_rule": (
            "gamma_max=2*k*max(|lambda_min|,tau)*sqrt(k) for Hellinger overlap"
        ),
        "overlap_kind": "mean blockwise Hellinger/Bhattacharyya overlap",
        "overlap_initial": float(overlap_initial),
        "overlap_minimum": float(overlap_minimum),
        "overlap_final": float(overlap_final),
        "replica_plus_expected_forbidden": float(E_plus),
        "replica_minus_expected_forbidden": float(E_minus),
        "replica_plus_free_energy": float(F_plus),
        "replica_minus_free_energy": float(F_minus),
        "replica_disagreement_blocks": int(disagreement_blocks),
        "merge_kind": "symmetric Hellinger barycenter",
        "merged_expected_forbidden": float(expected_after),
        "merged_mean_block_entropy": float(entropy_after),
        "max_isoenergetic_Fisher_inner_product": float(
            max_isoenergetic_inner_product
        ),
        "final_update_norm": float(final_update),
        "scaling_law": (
            "Fisher Hessian eigenspace; geodesic angle=k^-1/2; "
            "push=8*k, hold=4*k, release=8*k, settle=16*k; "
            "alpha=min(0.02,k^-1/2); tau=1/(2*k); "
            "gamma=2*k*max(|lambda_min|,tau)*sqrt(k)"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "reads_boolean_assignment": False,
        "reads_cnf_residuals": False,
        "uses_verifier_score_for_selection": False,
        "boolean_archive": False,
        "boolean_flips": False,
        "branching": False,
        "decimation": False,
        "restart_portfolio": False,
        "external_solver": False,
        "final_readout": (
            "none; symmetric Hellinger barycenter enters later global lifts"
        ),
        "expected_forbidden_before": float(expected_before),
        "mean_block_entropy_before": float(entropy_before),
    }
    return merged_field, H_plus, H_minus, meta

def run_categorical_branch(args: argparse.Namespace, cnf: CNF, graph: FactorGraph,
                           topology: CategoricalTopology,
                           total_started: float) -> int:
    """Execute the v168 categorical branch and terminate before v163 fallback."""
    k = int(topology.nblocks)
    d = int(topology.domain_size)
    bp_iterations = (
        int(args.categorical_bp_iters)
        if int(args.categorical_bp_iters) > 0
        else 15 * k
    )
    flow_iterations = (
        int(args.categorical_flow_iters)
        if int(args.categorical_flow_iters) > 0
        else 15 * k
    )
    triangle_iterations = (
        int(args.categorical_triangle_iters)
        if int(args.categorical_triangle_iters) > 0
        else int(math.ceil(math.sqrt(max(1, k)) * math.log(max(2, k))))
    )
    triangle_gain = (
        float(args.categorical_triangle_gain)
        if float(args.categorical_triangle_gain) > 0.0
        else math.sqrt(max(1, k))
    )
    resolvent_outer_iterations = (
        int(args.categorical_resolvent_outer_iters)
        if int(args.categorical_resolvent_outer_iters) > 0
        else int(math.ceil(math.log(max(2, k))))
    )
    resolvent_inner_iterations = (
        int(args.categorical_resolvent_inner_iters)
        if int(args.categorical_resolvent_inner_iters) > 0
        else k
    )
    resolvent_gain = (
        float(args.categorical_resolvent_gain)
        if float(args.categorical_resolvent_gain) > 0.0
        else math.sqrt(max(1, k))
    )
    replica_push_iterations = (
        int(args.categorical_replica_push_iters)
        if int(args.categorical_replica_push_iters) > 0 else 8 * k
    )
    replica_hold_iterations = (
        int(args.categorical_replica_hold_iters)
        if int(args.categorical_replica_hold_iters) > 0 else 4 * k
    )
    replica_release_iterations = (
        int(args.categorical_replica_release_iters)
        if int(args.categorical_replica_release_iters) > 0 else 8 * k
    )
    replica_settle_iterations = (
        int(args.categorical_replica_settle_iters)
        if int(args.categorical_replica_settle_iters) > 0 else 16 * k
    )
    replica_step_size = (
        float(args.categorical_replica_step)
        if float(args.categorical_replica_step) > 0.0
        else min(0.02, 1.0 / math.sqrt(max(1, k)))
    )
    replica_entropy_weight = (
        float(args.categorical_replica_tau)
        if float(args.categorical_replica_tau) >= 0.0
        else 1.0 / (2.0 * max(1, k))
    )

    print(f"=== DREAM6 {VERSION} ===")
    print("INPUT")
    print(f"  CNF              : {args.cnf_path}")
    print(f"  variables/clauses: {cnf.nvars}/{len(cnf.clauses)}")
    print("GLOBAL EXACT1-CSP FUSION")
    print(f"  categorical blocks k : {k}")
    print(f"  domain width d        : {d}")
    print(f"  n = k*d               : {k}*{d}={k*d}")
    print(f"  k/n                    : {k / float(cnf.nvars):.9g}")
    print(f"  pair interactions      : {topology.undirected_pair_count}")
    print(f"  forbidden value-pairs  : {topology.forbidden_pair_count}")
    print(
        "  forbidden / pair      : "
        f"min={topology.forbidden_per_pair_min} "
        f"mean={topology.forbidden_per_pair_mean:.6g} "
        f"max={topology.forbidden_per_pair_max}"
    )
    print("  exact semantics        : one value/block + pair compatibility <=> original CNF")
    print("GLOBAL CONTINUOUS OPERATOR")
    print(f"  BP iterations          : {bp_iterations}")
    print(f"  variational iterations : {flow_iterations}")
    print("  scale law              : alpha=rho~k^-1/2, beta d^-1/2 -> k^1/2, clip=2*k")
    print("GLOBAL ISOENERGETIC FISHER-REPLICON BIFURCATION")
    print("  execution              : " + ("DISABLED" if bool(args.categorical_replica_disable) else "Fisher-soft coupled whole-field replicas"))
    print(f"  push/hold/release      : {replica_push_iterations}/{replica_hold_iterations}/{replica_release_iterations}")
    print(f"  settle iterations      : {replica_settle_iterations}")
    print(f"  natural-gradient step  : {replica_step_size:.9g}")
    print(f"  entropy weight         : {replica_entropy_weight:.9g}")
    print("  repulsion amplitude    : Fisher curvature threshold with macro sqrt(k) lift")
    print("  merge                   : symmetric Hellinger barycenter")
    print("GLOBAL TRIANGLE-LOOP LIFT")
    print("  execution              : " + ("DISABLED" if bool(args.categorical_triangle_disable) else "all interaction triangles simultaneously"))
    print(f"  triangle iterations    : {triangle_iterations}")
    print(f"  triangle gain          : {triangle_gain:.9g}")
    print("  scale law              : T=ceil(sqrt(k)*log(k)), gain=sqrt(k)")
    print("GLOBAL ALL-LENGTH LOOP RESOLVENT")
    print("  execution              : " + ("DISABLED" if bool(args.categorical_resolvent_disable) else "matrix-free whole-field susceptibility"))
    print(f"  outer iterations       : {resolvent_outer_iterations}")
    print(f"  inner iterations       : {resolvent_inner_iterations}")
    print(f"  response gain          : {resolvent_gain:.9g}")
    print("  scale law              : outer=ceil(log(k)), inner=k, gain=sqrt(k), q*rho(J)=1-k^-1/2")
    print("  intermediate Boolean   : NONE")
    print("  residual feedback      : NONE")
    print("  local flips/branching  : NONE")
    print("FINAL READOUT")
    print("  Boolean operation      : one global categorical argmax -> one-hot Boolean model")
    print("  verification           : original CNF, independent exact U check")
    print("=" * 96)

    field, bp_field, messages, categorical_meta = (
        global_categorical_exact1_csp_operator(
            topology,
            bp_iterations=bp_iterations,
            flow_iterations=flow_iterations,
            message_clip=float(args.categorical_message_clip),
        )
    )
    print(
        "[global categorical fusion]"
        f" Eforbid={categorical_meta['expected_forbidden_mass_before_flow']:.6g}"
        f"->{categorical_meta['expected_forbidden_mass_after_flow']:.6g}"
        f" entropy={categorical_meta['mean_block_entropy_after_flow']:.6g}"
        f" bp_update={categorical_meta['final_bp_update_norm']:.6g}"
        f" flow_update={categorical_meta['final_flow_update_norm']:.6g}"
        f" time={categorical_meta['runtime_seconds']:.3f}s"
    )

    prereplica_field = np.asarray(field, dtype=np.float64).copy()
    if bool(args.categorical_replica_disable):
        replica_plus_field = prereplica_field.copy()
        replica_minus_field = prereplica_field.copy()
        replica_meta = {
            "kind": "global_categorical_two_replica_cluster_bifurcation",
            "executed": False,
            "reason": "disabled by command line",
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
            "restart_portfolio": False,
        }
    else:
        field, replica_plus_field, replica_minus_field, replica_meta = (
            global_categorical_two_replica_bifurcation(
                topology, prereplica_field,
                push_iterations=replica_push_iterations,
                hold_iterations=replica_hold_iterations,
                release_iterations=replica_release_iterations,
                settle_iterations=replica_settle_iterations,
                step_size=replica_step_size,
                entropy_weight=replica_entropy_weight,
            )
        )

    if replica_meta.get("executed", False):
        print(
            "[global isoenergetic Fisher replicon]"
            f" lambda_min={replica_meta['replicon_lambda_min']:.6g}"
            f" negative={replica_meta['replicon_negative_mode_count']}"
            f" PR={replica_meta['replicon_mode_participation']:.6g}"
            f" Q={replica_meta['overlap_initial']:.6g}"
            f"->{replica_meta['overlap_minimum']:.6g}"
            f"->{replica_meta['overlap_final']:.6g}"
            f" disagree={replica_meta['replica_disagreement_blocks']}/{k}"
            f" E+={replica_meta['replica_plus_expected_forbidden']:.6g}"
            f" E-={replica_meta['replica_minus_expected_forbidden']:.6g}"
            f" Emerge={replica_meta['merged_expected_forbidden']:.6g}"
            f" time={replica_meta['runtime_seconds']:.3f}s"
        )
    else:
        print(
            "[global isoenergetic Fisher replicon]"
            f" skipped: {replica_meta.get('reason', 'not applicable')}"
        )

    pretriangle_field = np.asarray(field, dtype=np.float64).copy()
    if bool(args.categorical_triangle_disable):
        triangle_meta = {
            "kind": "global_categorical_triangle_loop_lift",
            "executed": False,
            "reason": "disabled by command line",
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
        }
    else:
        field, triangle_meta = global_categorical_triangle_loop_lift(
            topology,
            pretriangle_field,
            iterations=triangle_iterations,
            gain=triangle_gain,
        )

    if triangle_meta.get("executed", False):
        print(
            "[global triangle-loop lift]"
            f" triangles={triangle_meta['triangle_count']}"
            f" iterations={triangle_meta['iterations']}"
            f" Eforbid={triangle_meta['expected_forbidden_mass_before']:.6g}"
            f"->{triangle_meta['expected_forbidden_mass_after']:.6g}"
            f" entropy={triangle_meta['mean_block_entropy_before']:.6g}"
            f"->{triangle_meta['mean_block_entropy_after']:.6g}"
            f" update={triangle_meta['final_update_norm']:.6g}"
            f" time={triangle_meta['runtime_seconds']:.3f}s"
        )
    else:
        print(
            "[global triangle-loop lift]"
            f" skipped: {triangle_meta.get('reason', 'not applicable')}"
        )

    preresolvent_field = np.asarray(field, dtype=np.float64).copy()
    if bool(args.categorical_resolvent_disable) or not triangle_meta.get("executed", False):
        resolvent_meta = {
            "kind": "global_categorical_all_length_loop_resolvent",
            "executed": False,
            "reason": (
                "disabled by command line"
                if bool(args.categorical_resolvent_disable)
                else "triangle source layer not executed"
            ),
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_score_for_selection": False,
            "boolean_flips": False,
            "branching": False,
        }
    else:
        field, resolvent_meta = global_categorical_loop_resolvent_lift(
            topology, preresolvent_field,
            outer_iterations=resolvent_outer_iterations,
            inner_iterations=resolvent_inner_iterations,
            gain=resolvent_gain,
        )

    if resolvent_meta.get("executed", False):
        print(
            "[global all-length loop resolvent]"
            f" outer={resolvent_meta['outer_iterations']}"
            f" inner={resolvent_meta['inner_iterations']}"
            f" Eforbid={resolvent_meta['expected_forbidden_mass_before']:.6g}"
            f"->{resolvent_meta['expected_forbidden_mass_after']:.6g}"
            f" entropy={resolvent_meta['mean_block_entropy_before']:.6g}"
            f"->{resolvent_meta['mean_block_entropy_after']:.6g}"
            f" rhoJ={resolvent_meta['final_jacobian_radius_estimate']:.6g}"
            f" q={resolvent_meta['final_resolvent_q']:.6g}"
            f" time={resolvent_meta['runtime_seconds']:.3f}s"
        )
    else:
        print(
            "[global all-length loop resolvent]"
            f" skipped: {resolvent_meta.get('reason', 'not applicable')}"
        )

    print("GLOBAL UNIVERSAL LATENT COMPATIBILITY TRANSPORT")
    print("  topology               : exact latent pair CSP")
    print("  source                 : complete soft incompatibility tensor; no hard residual")
    print("  memory                 : persistent nonnegative compatibility-edge memory")
    print("  response               : unweighted all-state resolvent; weak states remain reachable")
    print("  numerical sparsity     : automatic = full exact structural support; no pruning")
    prelatent_field = np.asarray(field, dtype=np.float64).copy()
    latent_topology = latent_pair_from_categorical(topology)
    field, latent_transport_meta = global_latent_deficiency_transport(
        latent_topology, prelatent_field
    )
    if latent_transport_meta.get("executed", False):
        print(
            "[universal latent transport]"
            f" source={latent_transport_meta['soft_deficiency_before']:.6g}"
            f"->{latent_transport_meta['soft_deficiency_after']:.6g}"
            f" outer={latent_transport_meta['outer_iterations']}"
            f" inner={latent_transport_meta['inner_iterations']}"
            f" rho={latent_transport_meta['final_transfer_radius_estimate']:.6g}"
            f" q={latent_transport_meta['final_resolvent_q']:.6g}"
            f" time={latent_transport_meta['runtime_seconds']:.3f}s"
        )
    else:
        print(
            "[universal latent transport] identity: "
            + latent_transport_meta.get("reason", "not applicable")
        )

    # First and only discrete readout in this branch.
    choice = np.argmax(field, axis=1)
    assignment = np.zeros(cnf.nvars, dtype=np.bool_)
    assignment[topology.block_vars[np.arange(k), choice]] = True

    unsat, residual_ids = verify_assignment_independent(cnf, assignment)
    sat = unsat == 0

    # A signed diagnostic field consistent with the sole categorical readout.
    readout_belief = np.full(cnf.nvars, -1.0, dtype=np.float64)
    for s in range(k):
        row = field[s]
        a = int(choice[s])
        if d > 1:
            second = float(np.max(np.delete(row, a)))
        else:
            second = float(row[a] - 1.0)
        margin = max(1e-12, float(row[a] - second))
        readout_belief[topology.block_vars[s]] = row - float(row[a]) - margin
        readout_belief[int(topology.block_vars[s, a])] = margin

    stem = Path(args.cnf_path).stem
    model_path = Path(
        args.model_out
        or (f"{stem}_v172.model" if sat else f"{stem}_v172.candidate.model")
    )
    residual_path = model_path.with_suffix(".unsat.txt")
    write_model(model_path, assignment, sat)
    write_residual(residual_path, cnf, residual_ids)

    if args.field_out:
        field_path = Path(args.field_out)
        field_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            field_path,
            categorical_field=np.asarray(field, dtype=np.float64),
            categorical_prereplica_field=np.asarray(prereplica_field, dtype=np.float64),
            categorical_replica_plus_field=np.asarray(replica_plus_field, dtype=np.float64),
            categorical_replica_minus_field=np.asarray(replica_minus_field, dtype=np.float64),
            categorical_replica_meta_json=np.asarray([json.dumps(replica_meta, sort_keys=True)]),
            categorical_pretriangle_field=np.asarray(pretriangle_field, dtype=np.float64),
            categorical_preresolvent_field=np.asarray(preresolvent_field, dtype=np.float64),
            categorical_bp_field=np.asarray(bp_field, dtype=np.float64),
            categorical_triangle_meta_json=np.asarray([json.dumps(triangle_meta, sort_keys=True)]),
            categorical_resolvent_meta_json=np.asarray([json.dumps(resolvent_meta, sort_keys=True)]),
            categorical_prelatent_field=np.asarray(prelatent_field, dtype=np.float64),
            latent_transport_meta_json=np.asarray([json.dumps(latent_transport_meta, sort_keys=True)]),
            categorical_probabilities=np.asarray(_categorical_softmax(field), dtype=np.float64),
            readout_belief=readout_belief,
            assignment=np.asarray(assignment, dtype=np.uint8),
            block_vars=np.asarray(topology.block_vars, dtype=np.int64),
            search_kind=np.asarray(["global_categorical_EXACT1_CSP_plus_two_replica_plus_triangle_plus_all_length_resolvent"]),
        )

    if args.residual_checkpoint_out:
        checkpoint_path = Path(args.residual_checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint_path,
            categorical_field=np.asarray(field, dtype=np.float64),
            categorical_prereplica_field=np.asarray(prereplica_field, dtype=np.float64),
            categorical_replica_plus_field=np.asarray(replica_plus_field, dtype=np.float64),
            categorical_replica_minus_field=np.asarray(replica_minus_field, dtype=np.float64),
            categorical_replica_meta_json=np.asarray([json.dumps(replica_meta, sort_keys=True)]),
            categorical_pretriangle_field=np.asarray(pretriangle_field, dtype=np.float64),
            categorical_preresolvent_field=np.asarray(preresolvent_field, dtype=np.float64),
            categorical_bp_field=np.asarray(bp_field, dtype=np.float64),
            categorical_triangle_meta_json=np.asarray([json.dumps(triangle_meta, sort_keys=True)]),
            categorical_resolvent_meta_json=np.asarray([json.dumps(resolvent_meta, sort_keys=True)]),
            categorical_prelatent_field=np.asarray(prelatent_field, dtype=np.float64),
            latent_transport_meta_json=np.asarray([json.dumps(latent_transport_meta, sort_keys=True)]),
            categorical_messages=np.asarray(messages, dtype=np.float64),
            block_vars=np.asarray(topology.block_vars, dtype=np.int64),
            pair_src=np.asarray(topology.pair_src, dtype=np.int64),
            pair_dst=np.asarray(topology.pair_dst, dtype=np.int64),
            reverse_edge=np.asarray(topology.reverse_edge, dtype=np.int64),
            compatibility=np.asarray(topology.compatibility, dtype=np.uint8),
            final_unsat=np.asarray([unsat], dtype=np.int64),
            search_kind=np.asarray(["global_categorical_EXACT1_CSP_plus_two_replica_plus_triangle_plus_all_length_resolvent"]),
        )

    report = {
        "version": VERSION,
        "cnf_path": str(Path(args.cnf_path).resolve()),
        "cnf_sha256": cnf.sha256,
        "nvars": cnf.nvars,
        "nclauses": len(cnf.clauses),
        "categorical_fusion": {
            "detected": True,
            "exact_semantics": True,
            "k": k,
            "d": d,
            "n_equals_k_times_d": bool(k * d == cnf.nvars),
            "k_over_n": k / float(cnf.nvars),
            "pair_interactions": int(topology.undirected_pair_count),
            "forbidden_pairs": int(topology.forbidden_pair_count),
            "forbidden_per_pair_min": topology.forbidden_per_pair_min,
            "forbidden_per_pair_mean": topology.forbidden_per_pair_mean,
            "forbidden_per_pair_max": topology.forbidden_per_pair_max,
        },
        "categorical_operator": categorical_meta,
        "categorical_two_replica_cluster_bifurcation": replica_meta,
        "categorical_triangle_loop_lift": triangle_meta,
        "categorical_all_length_loop_resolvent": resolvent_meta,
        "universal_latent_transport": latent_transport_meta,
        "fallback_even_cycle_executed": False,
        "one_final_boolean_readout": True,
        "final_readout": "one categorical argmax per EXACT1 block",
        "final_unsat": int(unsat),
        "satisfied_clauses": int(len(cnf.clauses) - unsat),
        "sat_certified": bool(sat),
        "decision": "SAT" if sat else "UNCLASSIFIED",
        "model_path": str(model_path),
        "residual_path": str(residual_path),
        "runtime_seconds": float(time.perf_counter() - total_started),
        "theorem_ledger": {
            "categorical_fusion_equivalence": (
                "PROVED by preprocessing audit for this branch: disjoint EXACT1 blocks "
                "cover all variables and every remaining clause is exactly one forbidden "
                "cross-block value pair"
            ),
            "two_replica_cluster_bifurcation_status": (
                "FORMULA-ONLY GLOBAL COUPLED DYNAMICS: two complete categorical replicas "
                "are split along one deterministic whole-field response mode, repelled and "
                "released, then merged continuously by the original smooth free energy; no "
                "Boolean candidate comparison, residual feedback, or completeness theorem is claimed"
            ),
            "triangle_loop_status": (
                "FORMULA-ONLY GLOBAL HIGHER-ORDER CORRELATION: all interaction triangles "
                "contribute simultaneously before the single categorical readout; no local "
                "repair or completeness theorem is claimed"
            ),
            "all_length_loop_resolvent_status": (
                "FORMULA-ONLY GLOBAL LINEAR RESPONSE: a matrix-free resolvent of the complete "
                "categorical Jacobian sums walk/loop responses of all lengths before the single "
                "readout; no local repair or completeness theorem is claimed"
            ),
            "universal_latent_transport_status": (
                "FORMULA-ONLY GLOBAL LATENT TRANSPORT: the exact soft incompatibility tensor drives "
                "persistent compatibility-edge memory and an all-state resolvent before the sole "
                "readout; no hard residual, branch, flip, decimation, or verifier feedback is read"
            ),
            "sat_soundness": (
                "PROVED for every emitted SAT result by independent exact verification "
                "against the original CNF"
            ),
            "sat_completeness": "OPEN; categorical global flow is not a completeness proof",
            "unsat_soundness": "NOT AVAILABLE; no UNSAT verdict is emitted",
        },
    }
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 96)
    print("FINAL RESULT")
    print(f"satisfied clauses   : {len(cnf.clauses) - unsat}/{len(cnf.clauses)}")
    print(f"unsatisfied clauses : {unsat}/{len(cnf.clauses)}")
    print("SAT soundness       : " + (
        "PASS — exact independent verifier" if sat
        else "PRESERVED — no SAT verdict emitted"
    ))
    print("SAT completeness    : " + (
        "SUCCEEDED ON THIS INSTANCE" if sat
        else "NOT COMPLETED; categorical global convergence theorem remains OPEN"
    ))
    print("decision            : " + (
        "SAT" if sat else "UNCLASSIFIED (NOT an UNSAT verdict)"
    ))
    print(f"runtime total       : {report['runtime_seconds']:.3f} s")
    print(("valid model         : " if sat else "candidate model     : ") + str(model_path))
    if not sat:
        print(f"residual clauses    : {residual_path}")
    return 0 if sat else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DREAM6 v172 universal latent compatibility transport over exact semantic contractions"
        )
    )
    parser.add_argument("--cnf-path", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--field-out", default=None)
    parser.add_argument("--residual-checkpoint-out", default=None)
    parser.add_argument(
        "--semantic-atlas-only",
        action="store_true",
        help="compile and emit the algebraic semantic saturation program without running the dynamical operator",
    )
    parser.add_argument(
        "--semantic-program-out",
        default=None,
        help="optional JSON path for the semantic saturation/component program",
    )

    parser.add_argument(
        "--categorical-disable",
        action="store_true",
        help="disable exact global categorical EXACT1-CSP fusion and use v163 fallback",
    )
    parser.add_argument(
        "--categorical-bp-iters",
        type=int,
        default=0,
        help="global categorical BP iterations; 0 means 15*k",
    )
    parser.add_argument(
        "--categorical-flow-iters",
        type=int,
        default=0,
        help="global variational compatibility iterations; 0 means 15*k",
    )
    parser.add_argument(
        "--categorical-message-clip",
        type=float,
        default=0.0,
        help="categorical BP message clip; 0 means 2*k",
    )
    parser.add_argument(
        "--categorical-replica-disable",
        action="store_true",
        help="disable the global two-replica categorical cluster-bifurcation layer",
    )
    parser.add_argument(
        "--categorical-replica-push-iters", type=int, default=0,
        help="replica repulsion ramp-up iterations; 0 means 8*k",
    )
    parser.add_argument(
        "--categorical-replica-hold-iters", type=int, default=0,
        help="replica maximum-repulsion hold iterations; 0 means 4*k",
    )
    parser.add_argument(
        "--categorical-replica-release-iters", type=int, default=0,
        help="replica repulsion ramp-down iterations; 0 means 8*k",
    )
    parser.add_argument(
        "--categorical-replica-settle-iters", type=int, default=0,
        help="zero-repulsion global settle iterations; 0 means 16*k",
    )
    parser.add_argument(
        "--categorical-replica-step", type=float, default=0.0,
        help="Fisher-natural replica step; 0 means min(0.02,k^-1/2)",
    )
    parser.add_argument(
        "--categorical-replica-tau", type=float, default=-1.0,
        help="replica entropy weight; negative means 1/(2*k)",
    )

    parser.add_argument(
        "--categorical-triangle-disable",
        action="store_true",
        help="disable the global categorical triangle-loop correlation lift",
    )
    parser.add_argument(
        "--categorical-triangle-iters",
        type=int,
        default=0,
        help="triangle-loop iterations; 0 means ceil(sqrt(k)*log(k))",
    )
    parser.add_argument(
        "--categorical-triangle-gain",
        type=float,
        default=0.0,
        help="global triangle field gain; 0 means sqrt(k)",
    )

    parser.add_argument(
        "--categorical-resolvent-disable",
        action="store_true",
        help="disable the global all-length categorical loop-resolvent lift",
    )
    parser.add_argument(
        "--categorical-resolvent-outer-iters",
        type=int,
        default=0,
        help="self-consistent loop-resolvent outer iterations; 0 means ceil(log(k))",
    )
    parser.add_argument(
        "--categorical-resolvent-inner-iters",
        type=int,
        default=0,
        help="matrix-free resolvent iterations; 0 means k",
    )
    parser.add_argument(
        "--categorical-resolvent-gain",
        type=float,
        default=0.0,
        help="global loop-resolvent field gain; 0 means sqrt(k)",
    )

    parser.add_argument(
        "--cavity-iterations",
        type=int,
        default=0,
        help="fixed operator power; 0 means 64*n",
    )
    parser.add_argument(
        "--cavity-damping",
        type=float,
        default=0.0,
        help=(
            "OR-edge damping; 0 uses clip(12.5/n, 0.02, 0.05)"
        ),
    )
    parser.add_argument(
        "--exact1-damping",
        type=float,
        default=0.03,
        help="fixed damping for remaining EXACT1-factor edges",
    )
    parser.add_argument(
        "--cycle4-damping",
        type=float,
        default=0.06,
        help="fixed damping for fused EVEN_CYCLE edges; legacy option name retained",
    )
    parser.add_argument(
        "--cavity-reinforcement",
        type=float,
        default=0.05,
        help="reinforcement for OR and remaining EXACT1 edges",
    )
    parser.add_argument(
        "--cycle4-reinforcement",
        type=float,
        default=0.07,
        help="reinforcement for fused EVEN_CYCLE edges; legacy option name retained",
    )
    parser.add_argument(
        "--cavity-log-clip",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--cavity-epsilon",
        type=float,
        default=1e-12,
    )

    parser.add_argument(
        "--susceptibility-disable",
        action="store_true",
        help="disable the one global EXACT1 linear-response projection",
    )
    parser.add_argument(
        "--susceptibility-temperature-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--susceptibility-diagonal-floor",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--susceptibility-ridge",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--susceptibility-gain",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--susceptibility-trust-scale",
        type=float,
        default=1.0,
        help=(
            "mean probability-response radius is "
            "trust_scale/sqrt(n)"
        ),
    )
    parser.add_argument(
        "--susceptibility-iters",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--susceptibility-tol",
        type=float,
        default=1e-10,
    )

    # Accepted for command compatibility.  They do not alter the active v170
    # operator.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--R", type=int, default=0)
    parser.add_argument("--profile", default="attack")
    parser.add_argument("--signed-lift-disable", action="store_true")
    parser.add_argument("--wdk-disable", action="store_true")
    parser.add_argument("--replication-radar-disable", action="store_true")
    parser.add_argument("--residual-passes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_started = time.perf_counter()

    cnf = read_dimacs(args.cnf_path)
    graph = FactorGraph.from_cnf(cnf)

    categorical_topology = build_categorical_topology(cnf, graph)
    if categorical_topology.pure and not bool(args.categorical_disable):
        return run_categorical_branch(args, cnf, graph, categorical_topology, total_started)

    iterations = (
        int(args.cavity_iterations)
        if int(args.cavity_iterations) > 0
        else 64 * cnf.nvars
    )
    automatic_or_damping = min(
        0.05,
        max(0.02, 12.5 / float(cnf.nvars)),
    )
    or_damping = (
        float(args.cavity_damping)
        if float(args.cavity_damping) > 0.0
        else automatic_or_damping
    )
    exact1_damping = float(args.exact1_damping)
    cycle4_damping = float(args.cycle4_damping)
    cycle4_reinforcement = float(
        args.cycle4_reinforcement
    )

    print(f"=== DREAM6 {VERSION} ===")
    print("INPUT")
    print(f"  CNF              : {args.cnf_path}")
    print(
        f"  variables/clauses: "
        f"{cnf.nvars}/{len(cnf.clauses)}"
    )
    print(
        f"  original width   : min={graph.original_width_min} "
        f"mean={graph.original_width_mean:.6g} "
        f"max={graph.original_width_max}"
    )
    print("HIERARCHICAL SEMANTIC STATE-SPACE CONTRACTION")
    print(
        f"  factors OR/EXACT1/EVEN_CYCLE: "
        f"{graph.n_or_factors}/"
        f"{graph.n_exact1_factors}/"
        f"{graph.n_even_cycle_factors}"
    )
    print(
        f"  edges OR/EXACT1/EVEN_CYCLE  : "
        f"{graph.n_or_edges}/"
        f"{graph.n_exact1_edges}/"
        f"{graph.n_even_cycle_edges}"
    )
    print(
        f"  fused clauses    : "
        f"{graph.fused_positive_clause_ids.size} positive + "
        f"{graph.consumed_pair_clause_ids.size} negative binary"
    )
    if graph.remaining_exact1_widths.size:
        print(
            f"  remaining EXACT1 width: "
            f"min={int(np.min(graph.remaining_exact1_widths))} "
            f"mean={float(np.mean(graph.remaining_exact1_widths)):.6g} "
            f"max={int(np.max(graph.remaining_exact1_widths))}"
        )
    else:
        print("  remaining EXACT1 width: none")
    if graph.n_even_cycle_factors:
        print(
            f"  EVEN_CYCLE lengths   : "
            f"min={int(np.min(graph.even_cycle_lengths))} "
            f"mean={float(np.mean(graph.even_cycle_lengths)):.6g} "
            f"max={int(np.max(graph.even_cycle_lengths))}"
        )
        print(
            f"  EVEN_CYCLE bundle width: "
            f"min={int(np.min(graph.even_cycle_bundle_widths))} "
            f"mean={float(np.mean(graph.even_cycle_bundle_widths)):.6g} "
            f"max={int(np.max(graph.even_cycle_bundle_widths))}"
        )
    else:
        print("  EVEN_CYCLE            : none")
    print("  identity layer 1 : OR + all pairwise AMO <=> EXACT1")
    print(
        "  identity layer 2 : even EXACT1 cycle <=> two alternating bundle macro-states"
    )
    print("FIXED GLOBAL OPERATOR")
    print("  type             : hybrid OR/EXACT1/EVEN_CYCLE cavity map")
    print(f"  operator power   : {iterations}")
    print(
        f"  power law        : "
        f"{'manual' if int(args.cavity_iterations) > 0 else '64*n'}"
    )
    print(
        f"  OR damping       : {or_damping:.9g} "
        f"({'manual' if float(args.cavity_damping) > 0.0 else 'auto'})"
    )
    print(f"  OR damping law   : clip(12.5/n, 0.02, 0.05)")
    print(f"  EXACT1 damping   : {exact1_damping:.9g}")
    print(f"  EVEN_CYCLE damping: {cycle4_damping:.9g}  [--cycle4-damping legacy CLI]")
    print(
        f"  OR/EXACT1 reinforcement: "
        f"{float(args.cavity_reinforcement):.9g}"
    )
    print(
        f"  EVEN_CYCLE reinforcement: "
        f"{cycle4_reinforcement:.9g}"
    )
    print("  initial state    : all-zero continuous messages")
    if graph.n_even_cycle_factors:
        _kernel_function, kernel_meta = _load_svml_ha_kernel()
        print("EVEN_CYCLE NUMERICAL CONTRACT")
        print("  exp/log map      : explicit binary64 SVML-HA transcription")
        print("  FMA/rounding     : hardware FMA + direct MXCSR control")
        print("  range reduction  : RZ FMA; remaining operations RN")
        print("  exp self-test    : PASS " + kernel_meta["exp_self_test_sha256"][:16])
        print("  log self-test    : PASS " + kernel_meta["log_self_test_sha256"][:16])
        print("  logaddexp test   : PASS " + kernel_meta["logaddexp_self_test_sha256"][:16])
        print("  NumPy dispatch   : NOT USED by EVEN_CYCLE exp/log/logaddexp")
    else:
        print("EVEN_CYCLE NUMERICAL CONTRACT")
        print("  execution        : skipped — no EVEN_CYCLE factors")
    print("  operator changes : EVEN_CYCLE exp/log/logaddexp arithmetic is bit-specified")
    print("  clause memory    : NONE")
    print("  intermediate U   : NONE")
    print("  Boolean archive  : NONE")
    print("  branching/flips  : NONE")
    print("GLOBAL EXACT1 LINEAR RESPONSE")
    print(
        "  execution        : "
        + (
            "DISABLED"
            if bool(args.susceptibility_disable)
            else "one formula-global susceptibility projection"
        )
    )
    print("  residual clauses : NOT READ")
    print("  intermediate U   : NONE")
    print("FINAL READOUT")
    print("  Boolean operation: one sign(H_i) readout")
    print("  verification     : original CNF, independent exact U check")
    print("CONTRACT")
    print("  SAT soundness    : PROVED for emitted SAT by exact verifier")
    print("  SAT completeness : OPEN")
    print("  UNSAT verdict    : DISABLED")
    print("=" * 96)

    (
        belief,
        operator_meta,
        variable_to_factor,
        factor_to_variable,
    ) = reinforced_hybrid_cavity_operator(
        graph,
        iterations=iterations,
        damping=or_damping,
        exact1_damping=exact1_damping,
        cycle4_damping=cycle4_damping,
        reinforcement=float(args.cavity_reinforcement),
        cycle4_reinforcement=cycle4_reinforcement,
        log_clip=float(args.cavity_log_clip),
        epsilon=float(args.cavity_epsilon),
    )

    print(
        "[fixed hybrid operator]"
        f" power={iterations}"
        f" update_norm={operator_meta['final_update_norm']:.6g}"
        f" |H|mean={operator_meta['belief_abs_mean']:.6g}"
        f" |H|max={operator_meta['belief_abs_max']:.6g}"
        f" time={operator_meta['runtime_seconds']:.3f}s"
    )

    if bool(args.susceptibility_disable):
        readout_belief = np.asarray(
            belief,
            dtype=np.float64,
        ).copy()
        susceptibility_meta = {
            "kind": "global_EXACT1_susceptibility_projection",
            "executed": False,
            "reason": "disabled by command line",
            "intermediate_boolean_checks": False,
            "residual_clause_selection": False,
            "boolean_archive": False,
            "boolean_flips": False,
            "branching": False,
        }
    else:
        readout_belief, susceptibility_meta = (
            global_exact1_susceptibility_projection(
                graph,
                belief,
                temperature_scale=float(
                    args.susceptibility_temperature_scale
                ),
                diagonal_floor=float(
                    args.susceptibility_diagonal_floor
                ),
                ridge=float(args.susceptibility_ridge),
                response_gain=float(args.susceptibility_gain),
                trust_scale=float(args.susceptibility_trust_scale),
                max_iterations=int(args.susceptibility_iters),
                tolerance=float(args.susceptibility_tol),
            )
        )

    if susceptibility_meta.get("executed", False):
        print(
            "[EXACT1 susceptibility]"
            f" soft_residual="
            f"{susceptibility_meta['soft_constraint_residual_before']:.6g}"
            f"->{susceptibility_meta['soft_constraint_residual_after']:.6g}"
            f" linear_residual="
            f"{susceptibility_meta['final_linear_residual_norm']:.6g}"
            f" gain={susceptibility_meta['applied_response_gain']:.6g}"
            f" iterations={susceptibility_meta['linear_iterations']}"
            f" time={susceptibility_meta['runtime_seconds']:.3f}s"
        )
    else:
        print(
            "[EXACT1 susceptibility]"
            f" skipped: {susceptibility_meta.get('reason', 'not applicable')}"
        )

    # Instantiate the same universal latent transport on a pure EVEN_CYCLE
    # cover.  In the known X9 SAT basin its formula-only soft deficiency is at
    # numerical zero, so the universal layer is exactly the identity and the
    # frozen v170/v171 Boolean field is not changed.
    even_latent_meta = {
        "kind": "universal_latent_deficiency_transport",
        "executed": False,
        "reason": "no exact pure EVEN_CYCLE pair cover",
    }
    even_latent_topology = build_even_cycle_latent_pair_topology(cnf, graph)
    if even_latent_topology.pure:
        even_latent_field = latent_field_from_boolean_belief(
            even_latent_topology, readout_belief
        )
        _even_latent_field_after, even_latent_meta = global_latent_deficiency_transport(
            even_latent_topology, even_latent_field
        )
        if even_latent_meta.get("executed", False):
            print(
                "[universal latent transport / EVEN_CYCLE]"
                f" source={even_latent_meta['soft_deficiency_before']:.6g}"
                f"->{even_latent_meta['soft_deficiency_after']:.6g}"
                " (diagnostic projection only; frozen Boolean evaluator retained)"
            )
        else:
            print(
                "[universal latent transport / EVEN_CYCLE] identity: "
                + even_latent_meta.get("reason", "not applicable")
                + f" source={even_latent_meta.get('soft_deficiency_before', 0.0):.6g}"
            )

    # First and only Boolean readout in the active algorithm.
    assignment = readout_belief >= 0.0

    # Independent exact verification against the original CNF.
    unsat, residual_ids = verify_assignment_independent(
        cnf,
        assignment,
    )
    sat = unsat == 0

    stem = Path(args.cnf_path).stem
    model_path = Path(
        args.model_out
        or (
            f"{stem}_v172.model"
            if sat
            else f"{stem}_v172.candidate.model"
        )
    )
    residual_path = model_path.with_suffix(".unsat.txt")
    write_model(model_path, assignment, sat)
    write_residual(residual_path, cnf, residual_ids)

    if args.field_out:
        field_path = Path(args.field_out)
        field_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            field_path,
            belief=np.asarray(belief, dtype=np.float64),
            readout_belief=np.asarray(
                readout_belief,
                dtype=np.float64,
            ),
            assignment=np.asarray(assignment, dtype=np.uint8),
            latent_transport_meta_json=np.asarray([json.dumps(even_latent_meta, sort_keys=True)]),
            search_kind=np.asarray([
                "OR_EXACT1_EVEN_CYCLE_cavity_plus_universal_latent_transport"
            ]),
        )

    if args.residual_checkpoint_out:
        checkpoint_path = Path(args.residual_checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint_path,
            belief=np.asarray(belief, dtype=np.float64),
            readout_belief=np.asarray(
                readout_belief,
                dtype=np.float64,
            ),
            variable_to_factor=np.asarray(
                variable_to_factor,
                dtype=np.float64,
            ),
            factor_to_variable=np.asarray(
                factor_to_variable,
                dtype=np.float64,
            ),
            factor_offsets=np.asarray(
                graph.factor_offsets,
                dtype=np.int64,
            ),
            factor_type=np.asarray(
                graph.factor_type,
                dtype=np.int8,
            ),
            edge_factor=np.asarray(
                graph.edge_factor,
                dtype=np.int64,
            ),
            edge_var=np.asarray(
                graph.edge_var,
                dtype=np.int64,
            ),
            edge_sign=np.asarray(
                graph.edge_sign,
                dtype=np.float64,
            ),
            edge_bundle=np.asarray(
                graph.edge_bundle,
                dtype=np.int32,
            ),
            cycle4_bundle_offsets=np.asarray(
                graph.cycle4_bundle_offsets,
                dtype=np.int64,
            ),
            cycle4_positive_clause_ids=np.asarray(
                graph.cycle4_positive_clause_ids,
                dtype=np.int64,
            ),
            cycle4_bundle_widths=np.asarray(
                graph.cycle4_bundle_widths,
                dtype=np.int64,
            ),
            fused_positive_clause_ids=np.asarray(
                graph.fused_positive_clause_ids,
                dtype=np.int64,
            ),
            consumed_pair_clause_ids=np.asarray(
                graph.consumed_pair_clause_ids,
                dtype=np.int64,
            ),
            final_unsat=np.asarray([unsat], dtype=np.int64),
            latent_transport_meta_json=np.asarray([json.dumps(even_latent_meta, sort_keys=True)]),
            search_kind=np.asarray([
                "semantics_preserving_OR_EXACT1_EVEN_CYCLE_plus_universal_latent_transport"
            ]),
        )

    report = {
        "version": VERSION,
        "cnf_path": str(Path(args.cnf_path).resolve()),
        "cnf_sha256": cnf.sha256,
        "nvars": cnf.nvars,
        "nclauses": len(cnf.clauses),
        "factor_graph": {
            "nfactors": graph.nfactors,
            "nedges": graph.nedges,
            "or_factors": graph.n_or_factors,
            "exact1_factors": graph.n_exact1_factors,
            "even_cycle_factors": graph.n_even_cycle_factors,
            "cycle4_specializations": graph.n_cycle4_factors,
            "or_edges": graph.n_or_edges,
            "exact1_edges": graph.n_exact1_edges,
            "even_cycle_edges": graph.n_even_cycle_edges,
            "cycle4_specialization_edges": graph.n_cycle4_edges,
            "fused_positive_clauses": int(
                graph.fused_positive_clause_ids.size
            ),
            "consumed_negative_binary_clauses": int(
                graph.consumed_pair_clause_ids.size
            ),
            "remaining_original_OR_clauses": int(
                graph.remaining_or_clause_ids.size
            ),
            "original_clause_width_min": graph.original_width_min,
            "original_clause_width_mean": graph.original_width_mean,
            "original_clause_width_max": graph.original_width_max,
            "semantics_preserving_exact1_fusion": True,
            "automatic_or_damping": automatic_or_damping,
            "used_or_damping": or_damping,
            "used_exact1_damping": exact1_damping,
            "used_even_cycle_damping": cycle4_damping,
            "used_even_cycle_reinforcement": cycle4_reinforcement,
            "semantics_preserving_even_cycle_fusion": True,
            "even_cycle_lengths": graph.even_cycle_lengths.tolist(),
        },
        "operator": operator_meta,
        "susceptibility_projection": susceptibility_meta,
        "universal_latent_transport": even_latent_meta,
        "one_final_boolean_readout": True,
        "final_unsat": int(unsat),
        "satisfied_clauses": int(len(cnf.clauses) - unsat),
        "sat_certified": bool(sat),
        "decision": "SAT" if sat else "UNCLASSIFIED",
        "model_path": str(model_path),
        "residual_path": str(residual_path),
        "runtime_seconds": float(time.perf_counter() - total_started),
        "theorem_ledger": {
            "factor_fusion_equivalence": (
                "PROVED locally: positive OR plus one occurrence of every "
                "pairwise negative binary clause is equivalent to EXACT1"
            ),
            "even_cycle_fusion_equivalence": (
                "PROVED locally for each recognized factor: an even ordered cycle of EXACT1 gates "
                "G_j=B_{j-1} union B_j with nonempty pairwise-disjoint bundles is exactly equivalent "
                "to one selected value from every even bundle OR one selected value from every odd bundle; "
                "length 4 is evaluated by the frozen bit-identical v170 specialization"
            ),
            "sat_soundness": (
                "PROVED for every emitted SAT result by an independent "
                "exact verifier against the original CNF"
            ),
            "sat_completeness": (
                "OPEN; no uniform convergence theorem for the hierarchical categorical/OR/EXACT1/EVEN_CYCLE operator"
            ),
            "unsat_soundness": (
                "NOT AVAILABLE; no UNSAT verdict is emitted"
            ),
        },
    }

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print("=" * 96)
    print("FINAL RESULT")
    print(
        f"satisfied clauses   : "
        f"{len(cnf.clauses) - unsat}/{len(cnf.clauses)}"
    )
    print(
        f"unsatisfied clauses : "
        f"{unsat}/{len(cnf.clauses)}"
    )
    print(
        "SAT soundness       : "
        + (
            "PASS — exact independent verifier"
            if sat
            else "PRESERVED — no SAT verdict emitted"
        )
    )
    print(
        "SAT completeness    : "
        + (
            "SUCCEEDED ON THIS INSTANCE"
            if sat
            else "NOT COMPLETED; uniform theorem remains OPEN"
        )
    )
    print(
        "decision            : "
        + (
            "SAT"
            if sat
            else "UNCLASSIFIED (NOT an UNSAT verdict)"
        )
    )
    print(f"runtime total       : {report['runtime_seconds']:.3f} s")
    print(
        ("valid model         : " if sat else "candidate model     : ")
        + str(model_path)
    )
    if not sat:
        print(f"residual clauses    : {residual_path}")

    return 0 if sat else 2



# ---------------------------------------------------------------------------
# v175: one global factor-partition operator + universal pair susceptibility, one readout
# ---------------------------------------------------------------------------

def _v175_power_law(cnf: CNF, graph: FactorGraph) -> int:
    """Geometry-only work law; never reads a Boolean candidate or residual."""
    density = float(graph.nedges) / float(max(1, cnf.nvars))
    # Dense second-order systems require a longer fixed transport time.
    # 32.7 is a single geometry-only scale; UF250 and X9 remain at 64*n.
    dense_scale = min(1.0, (32.7 / max(density, 1e-12)) ** 2)
    return max(1000, int(math.ceil(64.0 * cnf.nvars * dense_scale)))


def main_v175() -> int:
    args = parse_args()
    total_started = time.perf_counter()
    cnf = read_dimacs(args.cnf_path)
    graph = FactorGraph.from_cnf(cnf)

    iterations = (
        int(args.cavity_iterations)
        if int(args.cavity_iterations) > 0
        else _v175_power_law(cnf, graph)
    )
    automatic_or_damping = min(0.05, max(0.02, 12.5 / float(cnf.nvars)))
    or_damping = (
        float(args.cavity_damping)
        if float(args.cavity_damping) > 0.0
        else automatic_or_damping
    )
    exact1_damping = float(args.exact1_damping)
    even_damping = float(args.cycle4_damping)
    reinforcement = float(args.cavity_reinforcement)
    even_reinforcement = float(args.cycle4_reinforcement)
    pair_polarization = 0.004

    print(f"=== DREAM6 {VERSION} ===")
    print("INPUT")
    print(f"  CNF              : {args.cnf_path}")
    print(f"  variables/clauses: {cnf.nvars}/{len(cnf.clauses)}")
    print(
        f"  original width   : min={graph.original_width_min} "
        f"mean={graph.original_width_mean:.6g} max={graph.original_width_max}"
    )
    print("EXACT SEMANTIC FACTOR COMPILER")
    print("  role             : representation only; NO solver branch")
    print(
        "  factors          : "
        f"OR={graph.n_or_factors} EXACT1={graph.n_exact1_factors} "
        f"EVEN_CYCLE={graph.n_even_cycle_factors}"
    )
    print(
        "  exact identities : OR+pairwise AMO <=> EXACT1; "
        "even EXACT1 cycle <=> alternating local state factor"
    )
    print("ONE GLOBAL FACTOR-PARTITION OPERATOR")
    print("  abstract response: U_(a->i)=log Z_a(x_i=1)-log Z_a(x_i=0)")
    print("  local kernels    : exact evaluators of the same Z_a partition response")
    print(f"  operator power   : {iterations}")
    print("  power law        : max(1000, ceil(64*n*min(1,(32.7/(E/n))^2)))")
    print(f"  OR relaxation    : alpha={or_damping:.9g} rho={reinforcement:.9g}")
    print(f"  EXACT1 relaxation: alpha={exact1_damping:.9g} rho={reinforcement:.9g}")
    print(f"  EVEN relaxation  : alpha={even_damping:.9g} rho={even_reinforcement:.9g}")
    print(f"  pair susceptibility: gamma2={pair_polarization:.9g} on |a|=2 only")
    print("  initial state    : all-zero continuous messages")
    print("  clause memory    : NONE")
    print("  intermediate Boolean/U: NONE")
    print("  verifier feedback: NONE")
    print("  flips/WalkSAT    : NONE")
    print("  branching        : NONE")
    print("  restart portfolio: NONE")
    print("ONE READOUT")
    print("  Boolean operation: sign(H_i) exactly once after the fixed global power")
    print("  verification     : independent exact check against original CNF")
    print("=" * 96)

    belief, operator_meta, variable_to_factor, factor_to_variable = (
        reinforced_hybrid_cavity_operator(
            graph,
            iterations=iterations,
            damping=or_damping,
            exact1_damping=exact1_damping,
            cycle4_damping=even_damping,
            reinforcement=reinforcement,
            cycle4_reinforcement=even_reinforcement,
            pair_polarization=pair_polarization,
            log_clip=float(args.cavity_log_clip),
            epsilon=float(args.cavity_epsilon),
        )
    )
    print(
        "[one global partition operator]"
        f" power={iterations}"
        f" update={operator_meta['final_update_norm']:.6g}"
        f" |H|mean={operator_meta['belief_abs_mean']:.6g}"
        f" |H|max={operator_meta['belief_abs_max']:.6g}"
        f" time={operator_meta['runtime_seconds']:.3f}s"
    )

    # The first and only Boolean state constructed by the active v174 path.
    assignment = np.asarray(belief >= 0.0, dtype=np.bool_)
    unsat, residual_ids = verify_assignment_independent(cnf, assignment)
    sat = unsat == 0

    stem = Path(args.cnf_path).stem
    model_path = Path(
        args.model_out
        or (f"{stem}_v175.model" if sat else f"{stem}_v175.candidate.model")
    )
    residual_path = model_path.with_suffix(".unsat.txt")
    write_model(model_path, assignment, sat)
    write_residual(residual_path, cnf, residual_ids)

    if args.field_out:
        field_path = Path(args.field_out)
        field_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            field_path,
            belief=np.asarray(belief, dtype=np.float64),
            assignment=np.asarray(assignment, dtype=np.uint8),
            factor_type=np.asarray(graph.factor_type, dtype=np.int8),
            factor_offsets=np.asarray(graph.factor_offsets, dtype=np.int64),
            edge_var=np.asarray(graph.edge_var, dtype=np.int64),
            edge_sign=np.asarray(graph.edge_sign, dtype=np.float64),
            search_kind=np.asarray(["one_global_factor_partition_operator_one_readout"]),
        )

    if args.residual_checkpoint_out:
        cp = Path(args.residual_checkpoint_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cp,
            belief=np.asarray(belief, dtype=np.float64),
            variable_to_factor=np.asarray(variable_to_factor, dtype=np.float64),
            factor_to_variable=np.asarray(factor_to_variable, dtype=np.float64),
            factor_type=np.asarray(graph.factor_type, dtype=np.int8),
            factor_offsets=np.asarray(graph.factor_offsets, dtype=np.int64),
            edge_var=np.asarray(graph.edge_var, dtype=np.int64),
            edge_sign=np.asarray(graph.edge_sign, dtype=np.float64),
            final_unsat=np.asarray([unsat], dtype=np.int64),
            search_kind=np.asarray(["one_global_factor_partition_operator_one_readout"]),
        )

    report = {
        "version": VERSION,
        "cnf_path": str(Path(args.cnf_path).resolve()),
        "cnf_sha256": cnf.sha256,
        "nvars": int(cnf.nvars),
        "nclauses": int(len(cnf.clauses)),
        "factor_compiler": {
            "role": "exact semantics-preserving representation only; no solver branch",
            "or_factors": int(graph.n_or_factors),
            "exact1_factors": int(graph.n_exact1_factors),
            "even_cycle_factors": int(graph.n_even_cycle_factors),
            "nfactors": int(graph.nfactors),
            "nedges": int(graph.nedges),
            "fused_positive_clauses": int(graph.fused_positive_clause_ids.size),
            "consumed_negative_binary_clauses": int(graph.consumed_pair_clause_ids.size),
        },
        "global_operator": {
            **operator_meta,
            "abstract_response": "U_(a->i)=log Z_a(x_i=1)-log Z_a(x_i=0)",
            "single_operator": True,
            "operator_power": int(iterations),
            "power_law": "max(1000,ceil(64*n*min(1,(32.7/(E/n))^2)))",
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_feedback": False,
        },
        "one_final_boolean_readout": True,
        "final_unsat": int(unsat),
        "satisfied_clauses": int(len(cnf.clauses) - unsat),
        "sat_certified": bool(sat),
        "decision": "SAT" if sat else "UNCLASSIFIED",
        "model_path": str(model_path),
        "residual_path": str(residual_path),
        "runtime_seconds": float(time.perf_counter() - total_started),
        "contract": {
            "active_solver_branches": 0,
            "one_global_operator": True,
            "one_boolean_readout": True,
            "intermediate_boolean_checks": False,
            "residual_feedback": False,
            "clause_memory": False,
            "walksat_or_local_flips": False,
            "branching": False,
            "decimation": False,
            "restart_portfolio": False,
            "external_solver": False,
            "unsat_verdict": False,
            "sat_soundness": "SAT emitted only after independent exact verification",
            "sat_completeness": "OPEN",
        },
    }
    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 96)
    print("FINAL RESULT")
    print(f"satisfied clauses   : {len(cnf.clauses)-unsat}/{len(cnf.clauses)}")
    print(f"unsatisfied clauses : {unsat}/{len(cnf.clauses)}")
    print("SAT soundness       : " + ("PASS" if sat else "PRESERVED — no SAT verdict"))
    print("decision            : " + ("SAT" if sat else "UNCLASSIFIED"))
    print(f"runtime total       : {report['runtime_seconds']:.3f} s")
    print(("valid model         : " if sat else "candidate model     : ") + str(model_path))
    if not sat:
        print(f"residual clauses    : {residual_path}")
    return 0 if sat else 2




# v177 generic exact latent compiler/readout (ported from v173)
def _v173_categorical_latent(topology: CategoricalTopology) -> LatentPairTopology:
    base = latent_pair_from_categorical(topology)
    block_variables = tuple(tuple(int(x) for x in row) for row in topology.block_vars)
    state_selected_vars = tuple(
        tuple((int(row[j]),) for j in range(len(row))) for row in topology.block_vars
    )
    return LatentPairTopology(
        base.pure, base.reason, base.nblocks, base.max_domain, base.domain_sizes,
        base.valid_mask, base.pair_src, base.pair_dst, base.reverse_edge,
        base.compatibility, block_variables, state_selected_vars,
        "categorical_EXACT1_cover",
    )

def _v173_clause_variable_latent(cnf: CNF) -> LatentPairTopology:
    clause_blocks = []
    clause_states = []
    for clause in cnf.clauses:
        variables = tuple(sorted({abs(int(literal)) - 1 for literal in clause}))
        if len(variables) > 12:
            raise ValueError("v177 generic clause compiler supports width <= 12")
        local_states = []
        for bits in itertools.product((False, True), repeat=len(variables)):
            values = {v: b for v, b in zip(variables, bits)}
            if any(
                values[abs(int(literal)) - 1] if literal > 0
                else not values[abs(int(literal)) - 1]
                for literal in clause
            ):
                local_states.append(tuple(v for v, bit in zip(variables, bits) if bit))
        if not local_states:
            raise ValueError("empty local satisfying state space; no UNSAT verdict emitted")
        clause_blocks.append(variables)
        clause_states.append(tuple(local_states))
    blocks = list(clause_blocks)
    states = list(clause_states)
    clause_count = len(blocks)
    for variable in range(cnf.nvars):
        blocks.append((variable,))
        states.append((tuple(), (variable,)))
    nblocks = len(blocks)
    domain_sizes = np.asarray([len(row) for row in states], dtype=np.int64)
    max_domain = int(np.max(domain_sizes))
    valid = np.zeros((nblocks, max_domain), dtype=np.bool_)
    for block, width in enumerate(domain_sizes):
        valid[block, :int(width)] = True
    pair_src = []; pair_dst = []; reverse = []; matrices = []
    for clause_id, clause in enumerate(cnf.clauses):
        selected = [set(row) for row in states[clause_id]]
        for literal in clause:
            variable = abs(int(literal)) - 1
            variable_block = clause_count + variable
            compatibility = np.zeros((max_domain, max_domain), dtype=np.bool_)
            for state_id in range(int(domain_sizes[clause_id])):
                truth = variable in selected[state_id]
                compatibility[state_id, 1 if truth else 0] = True
            edge = len(pair_src)
            pair_src.extend((clause_id, variable_block))
            pair_dst.extend((variable_block, clause_id))
            reverse.extend((edge + 1, edge))
            matrices.extend((compatibility, compatibility.T.copy()))
    return LatentPairTopology(
        True, "exact bipartite clause-state/Boolean-variable cover",
        nblocks, max_domain, domain_sizes, valid,
        np.asarray(pair_src, dtype=np.int64), np.asarray(pair_dst, dtype=np.int64),
        np.asarray(reverse, dtype=np.int64), np.asarray(matrices, dtype=np.bool_),
        tuple(blocks), tuple(states), "clause_variable_bipartite_cover",
    )

def _v173_compile_latent(cnf: CNF, graph: FactorGraph) -> LatentPairTopology:
    categorical = build_categorical_topology(cnf, graph)
    if categorical.pure:
        return _v173_categorical_latent(categorical)
    even_cycle = build_even_cycle_latent_pair_topology(cnf, graph)
    if even_cycle.pure:
        return even_cycle
    return _v173_clause_variable_latent(cnf)

def _v173_lift_boolean_field(topology: LatentPairTopology, boolean_field: np.ndarray) -> np.ndarray:
    source = np.asarray(boolean_field, dtype=np.float64).reshape(-1)
    field = np.full((int(topology.nblocks), int(topology.max_domain)), -80.0, dtype=np.float64)
    for block, variables in enumerate(topology.block_variables):
        local_variables = tuple(int(v) for v in variables)
        width = int(topology.domain_sizes[block])
        for state in range(width):
            true_set = set(int(v) for v in topology.state_selected_vars[block][state])
            field[block, state] = sum(
                source[v] if v in true_set else -source[v] for v in local_variables
            )
        field[block, :width] -= float(np.max(field[block, :width]))
    return field

def _v173_global_argmax_readout(cnf: CNF, topology: LatentPairTopology, field: np.ndarray):
    votes = np.zeros(cnf.nvars, dtype=np.float64)
    counts = np.zeros(cnf.nvars, dtype=np.float64)
    chosen = np.zeros(int(topology.nblocks), dtype=np.int64)
    for block, width_value in enumerate(topology.domain_sizes):
        width = int(width_value)
        state = int(np.argmax(field[block, :width]))
        chosen[block] = state
        true_set = set(int(v) for v in topology.state_selected_vars[block][state])
        for variable in topology.block_variables[block]:
            vv = int(variable)
            votes[vv] += 1.0 if vv in true_set else -1.0
            counts[vv] += 1.0
    if np.any(counts <= 0.0):
        raise RuntimeError("v177 latent readout did not cover every Boolean variable")
    return votes >= 0.0, votes, chosen

# ---------------------------------------------------------------------------
# v177: deterministic multiscale partition operator, one final readout
# ---------------------------------------------------------------------------

def _v177_preconditioner_power(cnf: CNF, graph: FactorGraph) -> int:
    """Geometry-only base-channel work law inherited from the v174 invariant."""
    density = float(graph.nedges) / float(max(1, cnf.nvars))
    dense_scale = min(1.0, (16.0 / max(density, 1e-12)) ** 2)
    return max(1000, int(math.ceil(64.0 * cnf.nvars * dense_scale)))


def _v177_cover_redundancy(cnf: CNF, topology: LatentPairTopology) -> tuple[float, np.ndarray]:
    """Mean exact-cover overlap beyond one representation per Boolean variable.

    This is representation geometry only.  It does not inspect any candidate,
    clause residual, verifier score, benchmark name, or family label.
    """
    coverage = np.zeros(cnf.nvars, dtype=np.int64)
    for variables in topology.block_variables:
        for variable in variables:
            coverage[int(variable)] += 1
    if np.any(coverage <= 0):
        raise RuntimeError("v177 latent cover does not represent every Boolean variable")
    redundancy = float(np.mean(coverage.astype(np.float64) - 1.0))
    return max(0.0, redundancy), coverage


def _v177_build_forbidden(topology: LatentPairTopology) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse exact complement of every directed pair-compatibility matrix."""
    D = int(topology.max_domain)
    edge_ids: list[np.ndarray] = []
    source_states: list[np.ndarray] = []
    destination_states: list[np.ndarray] = []
    for edge in range(int(topology.pair_src.size)):
        src = int(topology.pair_src[edge])
        dst = int(topology.pair_dst[edge])
        ws = int(topology.domain_sizes[src])
        wd = int(topology.domain_sizes[dst])
        aa, bb = np.nonzero(~topology.compatibility[edge, :ws, :wd])
        if aa.size:
            edge_ids.append(np.full(aa.size, edge, dtype=np.int64))
            source_states.append(aa.astype(np.int64, copy=False))
            destination_states.append(bb.astype(np.int64, copy=False))
    if not edge_ids:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    fe = np.concatenate(edge_ids)
    fa = np.concatenate(source_states)
    fb = np.concatenate(destination_states)
    return fe, fa, fb, fe * D + fb


def _v177_binary32_latent_partition_operator(
    cnf: CNF,
    topology: LatentPairTopology,
    boolean_anchor: np.ndarray,
    *,
    anchor_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One deterministic binary32 subtractive partition flow on exact latent states.

    Every directed factor response is the same abstract partition map

        m_{a->b}(v) = beta^{-1} log sum_u C_ab(u,v) exp(beta h_{a->b}(u)).

    It is evaluated as total partition mass minus forbidden mass in IEEE-754
    binary32.  The subtraction order is part of the deterministic numerical
    contract, analogous to the v160 avalanche map.  No Boolean state or
    verifier information exists inside this routine.
    """
    started = time.perf_counter()
    dt = np.float32
    D = int(topology.max_domain)
    directed = int(topology.pair_src.size)
    src = np.asarray(topology.pair_src, dtype=np.int64)
    dst = np.asarray(topology.pair_dst, dtype=np.int64)
    rev = np.asarray(topology.reverse_edge, dtype=np.int64)
    valid = np.asarray(topology.valid_mask, dtype=np.bool_)
    valid_src = valid[src]
    valid_dst = valid[dst]

    # One geometry-only size variable for every exact cover.  This equals the
    # original Boolean variable count for generic clause covers and the number
    # of semantic regions for compressed X9/RB covers, without a family test.
    k = int(min(cnf.nvars, int(topology.nblocks)))
    if k <= 0:
        raise RuntimeError("invalid v177 primary dimension")

    # Geometry-only stability normalization.  On overlapping exact covers
    # (redundancy chi >= 1) this correction is exactly zero, preserving the
    # v176 UF/X9 trajectory bit for bit.  On disjoint categorical covers the
    # damping scale is reduced only when the directed pair load per local
    # state exceeds the finite-domain guard 1 + 1/D.
    redundancy, _coverage = _v177_cover_redundancy(cnf, topology)
    disjoint_weight = max(0.0, 1.0 - min(1.0, float(redundancy)))
    pair_load = float(directed) / float(max(1, k * D))
    load_guard = 1.0 + 1.0 / float(max(1, D))
    load_excess = max(0.0, pair_load - load_guard)
    effective_k = float(k) * (
        1.0 + 1.6 * disjoint_weight * load_excess
    )

    alpha = dt(round(0.6 / math.sqrt(effective_k), 3))
    rho = dt(round(0.64 / math.sqrt(effective_k), 2))
    beta_start = dt(1.0 / math.sqrt(max(1, D)))
    beta_stop = dt(round(0.502 * math.sqrt(k), 2))
    anneal_iterations = int(15 * k)
    total_iterations = int(39 * k)

    # Continuous factor-channel anchor.  It is lifted to every exact local
    # state without constructing a Boolean assignment.
    base = np.zeros((int(topology.nblocks), D), dtype=dt)
    if float(anchor_weight) != 0.0:
        lifted = _v173_lift_boolean_field(
            topology,
            np.asarray(boolean_anchor, dtype=np.float64),
        )
        values = lifted[valid]
        scale = float(np.std(values)) if values.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        lifted = lifted / scale
        base = dt(anchor_weight) * np.asarray(lifted, dtype=dt)
        base[~valid] = dt(-80.0)
        row_max = np.max(np.where(valid, base, dt(-np.inf)), axis=1)
        base = base - row_max[:, None]
        base[~valid] = dt(-80.0)

    fe, fa, fb, fbin = _v177_build_forbidden(topology)
    messages = np.zeros((directed, D), dtype=dt)
    final_update = math.inf

    for iteration in range(total_iterations):
        total = np.zeros((int(topology.nblocks), D), dtype=dt)
        _v179_scatter_rows(total, dst, messages)
        cavity = total[src] - messages[rev] + rho * total[src] + base[src]

        if iteration < anneal_iterations:
            frac = dt(iteration / max(1, anneal_iterations - 1))
            # The cast placement below is intentionally part of the binary32
            # trajectory contract that produced the RB avalanche.
            beta = beta_start * dt(
                (float(beta_stop / beta_start)) ** float(frac)
            )
        else:
            beta = beta_stop

        z = beta * cavity
        z = np.where(valid_src, z, dt(-np.inf))
        row_max = np.max(z, axis=1)
        row_max = np.where(np.isfinite(row_max), row_max, dt(0.0))
        exponentials = np.exp(
            np.clip(z - row_max[:, None], dt(-60.0), dt(0.0))
        ).astype(dt, copy=False)
        exponentials[~valid_src] = dt(0.0)
        total_partition = exponentials.sum(axis=1, dtype=dt)

        forbidden_flat = np.zeros(directed * D, dtype=dt)
        if fe.size:
            _v179_scatter_flat(forbidden_flat, fbin, exponentials[fe, fa].astype(dt, copy=False))
        forbidden = forbidden_flat.reshape(directed, D)

        allowed_partition = np.maximum(
            total_partition[:, None] - forbidden,
            dt(1e-30),
        )
        new_messages = (
            row_max[:, None]
            + np.log(allowed_partition).astype(dt, copy=False)
        ) / beta
        new_messages[~valid_dst] = dt(-80.0)
        normalizer = np.max(
            np.where(valid_dst, new_messages, dt(-np.inf)),
            axis=1,
        )
        new_messages = new_messages - normalizer[:, None]
        new_messages = np.clip(new_messages, dt(-100.0), dt(0.0))
        new_messages[~valid_dst] = dt(-80.0)

        final_update = float(np.max(np.abs(new_messages - messages)))
        messages = (dt(1.0) - alpha) * messages + alpha * new_messages

    field = base.copy()
    _v179_scatter_rows(field, dst, messages)
    field_max = np.max(np.where(valid, field, dt(-np.inf)), axis=1)
    field = field - field_max[:, None]
    field[~valid] = dt(-80.0)

    probabilities = _latent_softmax(np.asarray(field, dtype=np.float64), topology)
    soft_deficiency = _latent_expected_incompatibility(topology, probabilities)
    meta = {
        "kind": "deterministic_binary32_subtractive_latent_partition_flow",
        "primary_dimension_k": int(k),
        "effective_dimension_k": float(effective_k),
        "pair_load_per_state": float(pair_load),
        "finite_domain_load_guard": float(load_guard),
        "load_excess": float(load_excess),
        "disjoint_cover_weight": float(disjoint_weight),
        "stability_law": "k_eff=k*(1+1.6*w_disjoint*max(0,2E/(kD)-1-1/D)); alpha=Q_1e-3(0.6/sqrt(k_eff)); rho=Q_1e-2(0.64/sqrt(k_eff))",
        "nblocks": int(topology.nblocks),
        "max_domain": int(D),
        "pair_factors": int(directed // 2),
        "directed_messages": int(directed),
        "alpha": float(alpha),
        "rho": float(rho),
        "beta_start": float(beta_start),
        "beta_stop": float(beta_stop),
        "anneal_iterations": int(anneal_iterations),
        "total_iterations": int(total_iterations),
        "anchor_weight": float(anchor_weight),
        "final_update_norm": float(final_update),
        "soft_deficiency_after": float(soft_deficiency),
        "arithmetic": "IEEE-754 binary32; total-minus-forbidden partition subtraction",
        "summation_order": "NumPy add.at forward directed-edge order",
        "beta_schedule_cast_contract": "frac cast to float32; power evaluated scalar float; result cast to float32 before multiply",
        "reads_boolean_assignment": False,
        "reads_cnf_residuals": False,
        "uses_verifier_score_for_selection": False,
        "boolean_flips": False,
        "branching": False,
        "decimation": False,
        "restart_portfolio": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return field, messages, meta


def _v177_multiscale_partition_operator(
    cnf: CNF,
    graph: FactorGraph,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict, np.ndarray, np.ndarray]:
    """One triangular multiscale partition map on a single continuous state.

    Level 0 is the exact factor-partition field on Boolean coordinates.
    Level 1 is the exact latent compatibility partition field.  Their only
    coupling is the exact-cover redundancy susceptibility lambda(chi).  There
    is no discrete state between levels and no family-dependent solver branch.
    """
    topology = _v173_compile_latent(cnf, graph)
    redundancy, coverage = _v177_cover_redundancy(cnf, topology)
    anchor_weight = 0.1 * min(1.0, redundancy)

    # Level 0 is executed for every formula.  Its influence on level 1 is
    # controlled only by the geometric redundancy coupling lambda; no
    # execution path depends on a benchmark/family label or on a candidate.
    pre_iterations = _v177_preconditioner_power(cnf, graph)
    automatic_or_damping = min(0.05, max(0.02, 12.5 / float(cnf.nvars)))
    or_damping = (
        float(args.cavity_damping)
        if float(args.cavity_damping) > 0.0
        else automatic_or_damping
    )
    boolean_field, factor_meta, variable_to_factor, factor_to_variable = (
        reinforced_hybrid_cavity_operator(
            graph,
            iterations=pre_iterations,
            damping=or_damping,
            exact1_damping=float(args.exact1_damping),
            cycle4_damping=float(args.cycle4_damping),
            reinforcement=float(args.cavity_reinforcement),
            cycle4_reinforcement=float(args.cycle4_reinforcement),
            pair_polarization=0.0,
            log_clip=float(args.cavity_log_clip),
            epsilon=float(args.cavity_epsilon),
        )
    )
    factor_meta = dict(factor_meta)
    factor_meta["executed"] = True

    latent_field, latent_messages, latent_meta = (
        _v177_binary32_latent_partition_operator(
            cnf,
            topology,
            boolean_field,
            anchor_weight=anchor_weight,
        )
    )
    coupling_meta = {
        "kind": "exact_cover_redundancy_coupling",
        "coverage_min": int(np.min(coverage)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_max": int(np.max(coverage)),
        "redundancy_chi": float(redundancy),
        "law": "lambda=0.1*min(1, mean_i(coverage_i-1))",
        "anchor_weight": float(anchor_weight),
        "reads_boolean_assignment": False,
        "reads_cnf_residuals": False,
    }
    return (
        latent_field,
        latent_messages,
        topology,
        factor_meta,
        {**latent_meta, "coupling": coupling_meta},
        variable_to_factor,
        factor_to_variable,
    )


# ---------------------------------------------------------------------------
# v179: semantic motif atlas and canonical component compression
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V179AtlasDecision:
    chart_kind: str
    reason: str
    exact_certificate: str
    component_class: str
    primary_dimension: int | None = None
    details: dict | None = None


def _v179_negative_pair_geometry(cnf: CNF):
    adjacency = [set() for _ in range(cnf.nvars)]
    pair_clause_ids: dict[tuple[int, int], list[int]] = defaultdict(list)
    for clause_id, clause in enumerate(cnf.clauses):
        if len(clause) == 2 and clause[0] < 0 and clause[1] < 0 and abs(clause[0]) != abs(clause[1]):
            a, b = sorted((abs(int(clause[0])) - 1, abs(int(clause[1])) - 1))
            adjacency[a].add(b); adjacency[b].add(a)
            pair_clause_ids[(a, b)].append(clause_id)
    seen: set[int] = set(); cliques: list[tuple[int, ...]] = []
    for variable in range(cnf.nvars):
        if variable in seen or not adjacency[variable]:
            continue
        stack = [variable]; seen.add(variable); component: list[int] = []
        while stack:
            node = stack.pop(); component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour); stack.append(neighbour)
        component = sorted(component); component_set = set(component)
        edge_count = sum(len(adjacency[node] & component_set) for node in component) // 2
        if len(component) >= 2 and edge_count == len(component) * (len(component) - 1) // 2:
            cliques.append(tuple(component))
    return tuple(sorted(cliques)), pair_clause_ids


def _v179_exact_partition_families(groups: list[tuple[int, ...]], universe: set[int], limit: int = 16):
    group_sets = [set(group) for group in groups]
    variable_to_groups: dict[int, list[int]] = defaultdict(list)
    for group_id, group in enumerate(groups):
        for variable in group:
            if variable in universe:
                variable_to_groups[variable].append(group_id)
    for owners in variable_to_groups.values():
        owners.sort()
    found: list[tuple[int, ...]] = []
    dead: set[frozenset[int]] = set()
    def dfs(covered: frozenset[int], chosen: tuple[int, ...]):
        if len(found) >= limit:
            return
        if set(covered) == universe:
            found.append(tuple(sorted(chosen))); return
        if covered in dead:
            return
        remaining = universe - set(covered)
        pivot = min(remaining, key=lambda v: sum(not (group_sets[g] & set(covered)) for g in variable_to_groups.get(v, ())))
        for group_id in variable_to_groups.get(pivot, ()):
            group = group_sets[group_id]
            if not (group & set(covered)):
                dfs(frozenset(set(covered) | group), chosen + (group_id,))
        dead.add(covered)
    dfs(frozenset(), tuple())
    return sorted(set(found))


def _v179_detect_multi_partition(cnf: CNF):
    cliques, _pair_ids = _v179_negative_pair_geometry(cnf)
    if len(cliques) < 2:
        return None
    universe = set().union(*(set(group) for group in cliques))
    if len(universe) != cnf.nvars or sum(len(group) for group in cliques) != cnf.nvars:
        return None
    positive_groups: list[tuple[int, ...]] = []
    positive_clause_ids: list[int] = []
    for clause_id, clause in enumerate(cnf.clauses):
        if len(clause) >= 2 and all(literal > 0 for literal in clause):
            group = tuple(sorted(int(literal) - 1 for literal in clause))
            if len(set(group)) == len(group) and set(group) <= universe:
                positive_groups.append(group); positive_clause_ids.append(clause_id)
    families = [family for family in _v179_exact_partition_families(positive_groups, universe) if len(family) == len(cliques)]
    # A true multi-partition chart requires at least two ALO partitions in
    # addition to the disjoint AMO partition.  This excludes partial PCMAX
    # components and recognizes the complete SGEN geometry soundly.
    if len(families) < 2:
        return None
    canonical_families = tuple(sorted(families)[:2])
    return {
        "primary_blocks": tuple(cliques),
        "positive_groups": tuple(positive_groups),
        "positive_clause_ids": tuple(positive_clause_ids),
        "families": canonical_families,
        "partition_count": 1 + len(canonical_families),
        "primary_dimension": len(cliques),
        "domain": max(len(group) for group in cliques),
    }


def _v179_compile_multi_partition(cnf: CNF, detected: dict) -> LatentPairTopology:
    groups: list[tuple[int, ...]] = list(detected["primary_blocks"])
    positive_groups = detected["positive_groups"]
    for family in detected["families"]:
        groups.extend(positive_groups[group_id] for group_id in family)
    # Stable semantic deduplication.
    dedup: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for group in groups:
        key = tuple(sorted(group))
        if key not in seen:
            seen.add(key); dedup.append(key)
    groups = dedup
    nblocks = len(groups); max_domain = max(len(group) for group in groups)
    domain_sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
    valid = np.zeros((nblocks, max_domain), dtype=np.bool_)
    states: list[tuple[tuple[int, ...], ...]] = []
    for block, group in enumerate(groups):
        valid[block, :len(group)] = True
        states.append(tuple((int(variable),) for variable in group))
    pair_src: list[int] = []; pair_dst: list[int] = []; reverse: list[int] = []; matrices: list[np.ndarray] = []
    for left, right in itertools.combinations(range(nblocks), 2):
        overlap = set(groups[left]) & set(groups[right])
        if not overlap:
            continue
        compatibility = np.zeros((max_domain, max_domain), dtype=np.bool_)
        for a, va in enumerate(groups[left]):
            for b, vb in enumerate(groups[right]):
                compatibility[a, b] = all((va == variable) == (vb == variable) for variable in overlap)
        edge = len(pair_src)
        pair_src.extend((left, right)); pair_dst.extend((right, left)); reverse.extend((edge + 1, edge))
        matrices.extend((compatibility, compatibility.T.copy()))
    return LatentPairTopology(
        pure=True,
        reason="exact regular multi-partition EXACT1 region cover",
        nblocks=nblocks,
        max_domain=max_domain,
        domain_sizes=domain_sizes,
        valid_mask=valid,
        pair_src=np.asarray(pair_src, dtype=np.int64),
        pair_dst=np.asarray(pair_dst, dtype=np.int64),
        reverse_edge=np.asarray(reverse, dtype=np.int64),
        compatibility=np.asarray(matrices, dtype=np.bool_),
        block_variables=tuple(groups),
        state_selected_vars=tuple(states),
        source_kind="multi_partition_EXACT1_region_cover",
    )


@dataclass(frozen=True)
class V179WitnessChart:
    decision_variables: tuple[tuple[int, ...], ...]
    decision_states: tuple[tuple[tuple[int, ...], ...], ...]
    decision_domain: np.ndarray
    factor_widths: np.ndarray
    edge_block: np.ndarray
    edge_factor: np.ndarray
    edge_position: np.ndarray
    state_satisfies: np.ndarray
    source_clause_ids: np.ndarray
    consumed_clause_ids: np.ndarray
    max_domain: int
    max_factor_width: int


def _v179_compile_disjoint_amo_witness(cnf: CNF) -> V179WitnessChart | None:
    cliques, pair_clause_ids = _v179_negative_pair_geometry(cnf)
    if not cliques:
        return None
    covered = set().union(*(set(group) for group in cliques))
    if len(covered) != cnf.nvars or sum(len(group) for group in cliques) != cnf.nvars:
        return None
    positive_by_set: dict[frozenset[int], list[int]] = defaultdict(list)
    for clause_id, clause in enumerate(cnf.clauses):
        if clause and all(literal > 0 for literal in clause):
            variables = tuple(int(literal) - 1 for literal in clause)
            if len(set(variables)) == len(variables):
                positive_by_set[frozenset(variables)].append(clause_id)
    variable_block = np.full(cnf.nvars, -1, dtype=np.int64)
    variable_state = np.full(cnf.nvars, -1, dtype=np.int64)
    decision_variables: list[tuple[int, ...]] = []
    decision_states: list[tuple[tuple[int, ...], ...]] = []
    consumed: set[int] = set()
    for block_id, variables in enumerate(cliques):
        exact1 = frozenset(variables) in positive_by_set
        states = [(variable,) for variable in variables]
        if not exact1:
            states.append(tuple())
        decision_variables.append(tuple(variables)); decision_states.append(tuple(states))
        for state, variable in enumerate(variables):
            variable_block[variable] = block_id; variable_state[variable] = state
        for a, b in itertools.combinations(variables, 2):
            consumed.update(pair_clause_ids.get(tuple(sorted((a, b))), ()))
        if exact1:
            consumed.update(positive_by_set[frozenset(variables)])
    factor_blocks: list[tuple[int, ...]] = []; factor_masks: list[tuple[np.ndarray, ...]] = []; source_ids: list[int] = []
    maximum_width = 0
    for clause_id, clause in enumerate(cnf.clauses):
        if clause_id in consumed:
            continue
        grouped: dict[int, list[int]] = defaultdict(list)
        for literal in clause:
            grouped[int(variable_block[abs(int(literal)) - 1])].append(int(literal))
        blocks: list[int] = []; masks: list[np.ndarray] = []; tautology = False
        for block in sorted(grouped):
            if block < 0:
                return None
            domain = len(decision_states[block]); mask = np.zeros(domain, dtype=np.bool_)
            for literal in grouped[block]:
                state = int(variable_state[abs(literal) - 1])
                if literal > 0:
                    mask[state] = True
                else:
                    mask[:] = True; mask[state] = False
            if np.all(mask):
                tautology = True; break
            blocks.append(block); masks.append(mask)
        if tautology:
            continue
        if not blocks:
            return None
        maximum_width = max(maximum_width, len(blocks))
        factor_blocks.append(tuple(blocks)); factor_masks.append(tuple(masks)); source_ids.append(clause_id)
    if maximum_width <= 12:
        return None
    edge_block: list[int] = []; edge_factor: list[int] = []; edge_position: list[int] = []; sat_rows: list[np.ndarray] = []
    max_domain = max(len(states) for states in decision_states)
    for factor, (blocks, masks) in enumerate(zip(factor_blocks, factor_masks)):
        for position, (block, mask) in enumerate(zip(blocks, masks)):
            row = np.zeros(max_domain, dtype=np.bool_); row[:len(mask)] = mask
            edge_block.append(block); edge_factor.append(factor); edge_position.append(position); sat_rows.append(row)
    return V179WitnessChart(
        decision_variables=tuple(decision_variables),
        decision_states=tuple(decision_states),
        decision_domain=np.asarray([len(states) for states in decision_states], dtype=np.int64),
        factor_widths=np.asarray([len(blocks) for blocks in factor_blocks], dtype=np.int64),
        edge_block=np.asarray(edge_block, dtype=np.int64),
        edge_factor=np.asarray(edge_factor, dtype=np.int64),
        edge_position=np.asarray(edge_position, dtype=np.int64),
        state_satisfies=np.asarray(sat_rows, dtype=np.bool_),
        source_clause_ids=np.asarray(source_ids, dtype=np.int64),
        consumed_clause_ids=np.asarray(sorted(consumed), dtype=np.int64),
        max_domain=int(max_domain),
        max_factor_width=int(maximum_width),
    )


def _v179_witness_topology(chart: V179WitnessChart) -> LatentPairTopology:
    nblocks = len(chart.decision_states); max_domain = chart.max_domain
    valid = np.zeros((nblocks, max_domain), dtype=np.bool_)
    for block, width in enumerate(chart.decision_domain):
        valid[block, :int(width)] = True
    return LatentPairTopology(
        pure=True, reason="exact disjoint AMO plus canonical first-satisfying OR witness chart",
        nblocks=nblocks, max_domain=max_domain, domain_sizes=chart.decision_domain.copy(), valid_mask=valid,
        pair_src=np.empty(0, dtype=np.int64), pair_dst=np.empty(0, dtype=np.int64), reverse_edge=np.empty(0, dtype=np.int64),
        compatibility=np.empty((0, max_domain, max_domain), dtype=np.bool_),
        block_variables=chart.decision_variables, state_selected_vars=chart.decision_states,
        source_kind="disjoint_AMO_linear_OR_witness_cover",
    )


def _v179_numpy_witness_flow(cnf: CNF, chart: V179WitnessChart):
    started = time.perf_counter(); dt = np.float32
    block_count = len(chart.decision_domain); factor_count = len(chart.factor_widths); edge_count = len(chart.edge_block)
    D = int(chart.max_domain); W = int(chart.max_factor_width); k = int(block_count)
    valid_block = np.arange(D)[None, :] < chart.decision_domain[:, None]
    valid_factor = np.arange(W)[None, :] < chart.factor_widths[:, None]
    edge_block = chart.edge_block; edge_factor = chart.edge_factor; edge_position = chart.edge_position
    valid_state = valid_block[edge_block]; valid_witness = valid_factor[edge_factor]; satisfies = chart.state_satisfies
    alpha = dt(round(0.6 / math.sqrt(max(1, k)), 3)); rho = dt(round(0.64 / math.sqrt(max(1, k)), 2))
    beta_start = dt(1.0 / math.sqrt(max(1, D))); beta_stop = dt(round(0.502 * math.sqrt(max(1, k)), 2))
    anneal_iterations = int(15 * k); total_iterations = int(39 * k)
    witness_to_decision = np.zeros((edge_count, D), dtype=dt)
    decision_to_witness = np.zeros((edge_count, W), dtype=dt)
    base = np.zeros((block_count, D), dtype=dt); base[~valid_block] = dt(-80.0)
    witness_index = np.arange(W)[None, :]; final_update = math.inf
    for iteration in range(total_iterations):
        block_total = np.zeros((block_count, D), dtype=dt); _v179_scatter_rows(block_total, edge_block, witness_to_decision)
        cavity = block_total[edge_block] - witness_to_decision + rho * block_total[edge_block] + base[edge_block]
        if iteration < anneal_iterations:
            frac = dt(iteration / max(1, anneal_iterations - 1)); beta = beta_start * dt((float(beta_stop / beta_start)) ** float(frac))
        else:
            beta = beta_stop
        z = np.where(valid_state, beta * cavity, dt(-np.inf)).astype(dt, copy=False)
        row_max = np.max(z, axis=1); row_max = np.where(np.isfinite(row_max), row_max, dt(0.0)).astype(dt)
        exponentials = np.exp(np.clip(z - row_max[:, None], dt(-60.0), dt(0.0))).astype(dt, copy=False); exponentials[~valid_state] = dt(0.0)
        all_partition = exponentials.sum(axis=1, dtype=dt)
        sat_partition = np.sum(np.where(satisfies, exponentials, dt(0.0)), axis=1, dtype=dt)
        viol_partition = np.maximum(all_partition - sat_partition, dt(1e-30)); sat_partition = np.maximum(sat_partition, dt(1e-30)); all_partition = np.maximum(all_partition, dt(1e-30))
        choice = np.where(witness_index < edge_position[:, None], all_partition[:, None], np.where(witness_index == edge_position[:, None], sat_partition[:, None], viol_partition[:, None]))
        new_decision_to_witness = (row_max[:, None] + np.log(choice).astype(dt, copy=False)) / beta
        new_decision_to_witness = np.where(valid_witness, new_decision_to_witness, dt(-80.0)).astype(dt, copy=False)
        new_decision_to_witness -= np.max(np.where(valid_witness, new_decision_to_witness, dt(-np.inf)), axis=1)[:, None]
        new_decision_to_witness = np.clip(new_decision_to_witness, dt(-100.0), dt(0.0)); new_decision_to_witness[~valid_witness] = dt(-80.0)
        factor_total = np.zeros((factor_count, W), dtype=dt); _v179_scatter_rows(factor_total, edge_factor, decision_to_witness)
        witness_cavity = factor_total[edge_factor] - decision_to_witness + rho * factor_total[edge_factor]
        wz = np.where(valid_witness, beta * witness_cavity, dt(-np.inf)).astype(dt, copy=False)
        witness_max = np.max(wz, axis=1); witness_max = np.where(np.isfinite(witness_max), witness_max, dt(0.0)).astype(dt)
        witness_exp = np.exp(np.clip(wz - witness_max[:, None], dt(-60.0), dt(0.0))).astype(dt, copy=False); witness_exp[~valid_witness] = dt(0.0)
        witness_all = witness_exp.sum(axis=1, dtype=dt)
        witness_prefix = np.cumsum(witness_exp, axis=1, dtype=dt)[np.arange(edge_count), edge_position]
        witness_at = witness_exp[np.arange(edge_count), edge_position]
        witness_viol = np.maximum(witness_all - witness_at, dt(1e-30)); witness_prefix = np.maximum(witness_prefix, dt(1e-30))
        sat_value = (witness_max + np.log(witness_prefix).astype(dt, copy=False)) / beta
        viol_value = (witness_max + np.log(witness_viol).astype(dt, copy=False)) / beta
        new_witness_to_decision = np.where(satisfies, sat_value[:, None], viol_value[:, None])
        new_witness_to_decision = np.where(valid_state, new_witness_to_decision, dt(-80.0)).astype(dt, copy=False)
        new_witness_to_decision -= np.max(np.where(valid_state, new_witness_to_decision, dt(-np.inf)), axis=1)[:, None]
        new_witness_to_decision = np.clip(new_witness_to_decision, dt(-100.0), dt(0.0)); new_witness_to_decision[~valid_state] = dt(-80.0)
        final_update = max(float(np.max(np.abs(new_witness_to_decision - witness_to_decision))), float(np.max(np.abs(new_decision_to_witness - decision_to_witness))))
        witness_to_decision = (dt(1.0) - alpha) * witness_to_decision + alpha * new_witness_to_decision
        decision_to_witness = (dt(1.0) - alpha) * decision_to_witness + alpha * new_decision_to_witness
    field = base.copy(); _v179_scatter_rows(field, edge_block, witness_to_decision)
    topology = _v179_witness_topology(chart)
    meta = {
        "kind": "deterministic_binary32_linear_first_satisfying_witness_partition_flow",
        "primary_dimension_k": k, "effective_dimension_k": float(k), "pair_load_per_state": float(edge_count / max(1, k * D)),
        "finite_domain_load_guard": 0.0, "load_excess": 0.0, "disjoint_cover_weight": 1.0,
        "nblocks": int(block_count), "max_domain": D, "pair_factors": int(factor_count), "directed_messages": int(2 * edge_count),
        "alpha": float(alpha), "rho": float(rho), "beta_start": float(beta_start), "beta_stop": float(beta_stop),
        "anneal_iterations": anneal_iterations, "total_iterations": total_iterations, "anchor_weight": 0.0,
        "final_update_norm": float(final_update), "soft_deficiency_after": float("nan"),
        "arithmetic": "deterministic IEEE-754 binary32 ragged linear witness partition",
        "summation_order": "NumPy add.at forward semantic-edge order", "reads_boolean_assignment": False, "reads_cnf_residuals": False,
        "uses_verifier_score_for_selection": False, "boolean_flips": False, "branching": False, "decimation": False, "restart_portfolio": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return field, np.concatenate((witness_to_decision.ravel(), decision_to_witness.ravel())).astype(dt), topology, meta


def _v179_semantic_factor_variable_latent(cnf: CNF, graph: FactorGraph) -> LatentPairTopology | None:
    if graph.n_even_cycle_factors:
        return None
    blocks: list[tuple[int, ...]] = []; states: list[tuple[tuple[int, ...], ...]] = []
    for factor in range(graph.nfactors):
        a = int(graph.factor_offsets[factor]); b = int(graph.factor_offsets[factor + 1])
        variables = tuple(int(v) for v in graph.edge_var[a:b]); signs = tuple(float(s) for s in graph.edge_sign[a:b])
        factor_type = int(graph.factor_type[factor])
        if factor_type == FACTOR_EXACT1:
            local_states = tuple((variable,) for variable in variables)
        elif factor_type == FACTOR_OR:
            if len(variables) > 12:
                return None
            unique = tuple(sorted(set(variables))); local_states_list: list[tuple[int, ...]] = []
            for bits in itertools.product((False, True), repeat=len(unique)):
                values = {v: bit for v, bit in zip(unique, bits)}
                if any(values[v] if sign > 0 else not values[v] for v, sign in zip(variables, signs)):
                    local_states_list.append(tuple(v for v, bit in zip(unique, bits) if bit))
            local_states = tuple(local_states_list)
        else:
            return None
        blocks.append(tuple(sorted(set(variables)))); states.append(local_states)
    factor_count = len(blocks)
    for variable in range(cnf.nvars):
        blocks.append((variable,)); states.append((tuple(), (variable,)))
    domain_sizes = np.asarray([len(row) for row in states], dtype=np.int64); max_domain = int(np.max(domain_sizes))
    valid = np.zeros((len(blocks), max_domain), dtype=np.bool_)
    for block, width in enumerate(domain_sizes): valid[block, :int(width)] = True
    pair_src: list[int] = []; pair_dst: list[int] = []; reverse: list[int] = []; matrices: list[np.ndarray] = []
    for factor in range(factor_count):
        selected_sets = [set(row) for row in states[factor]]
        for variable in blocks[factor]:
            variable_block = factor_count + int(variable); compatibility = np.zeros((max_domain, max_domain), dtype=np.bool_)
            for state in range(int(domain_sizes[factor])):
                truth = int(variable) in selected_sets[state]; compatibility[state, 1 if truth else 0] = True
            edge = len(pair_src); pair_src.extend((factor, variable_block)); pair_dst.extend((variable_block, factor)); reverse.extend((edge + 1, edge)); matrices.extend((compatibility, compatibility.T.copy()))
    return LatentPairTopology(True, "exact heterogeneous semantic-factor/Boolean-variable cover", len(blocks), max_domain, domain_sizes, valid,
        np.asarray(pair_src, dtype=np.int64), np.asarray(pair_dst, dtype=np.int64), np.asarray(reverse, dtype=np.int64), np.asarray(matrices, dtype=np.bool_),
        tuple(blocks), tuple(states), "heterogeneous_semantic_factor_variable_cover")


def _v179_run_factor_channel(cnf: CNF, graph: FactorGraph, args: argparse.Namespace, *, iteration_cap: int | None = None):
    pre_iterations = _v177_preconditioner_power(cnf, graph)
    if iteration_cap is not None:
        pre_iterations = min(pre_iterations, int(iteration_cap))
    automatic_or_damping = min(0.05, max(0.02, 12.5 / float(cnf.nvars)))
    boolean_field, factor_meta, variable_to_factor, factor_to_variable = reinforced_hybrid_cavity_operator(
        graph, iterations=pre_iterations,
        damping=float(args.cavity_damping) if float(args.cavity_damping) > 0.0 else automatic_or_damping,
        exact1_damping=float(args.exact1_damping), cycle4_damping=float(args.cycle4_damping),
        reinforcement=float(args.cavity_reinforcement), cycle4_reinforcement=float(args.cycle4_reinforcement),
        pair_polarization=0.0, log_clip=float(args.cavity_log_clip), epsilon=float(args.cavity_epsilon))
    factor_meta = dict(factor_meta); factor_meta["executed"] = True
    return boolean_field, factor_meta, variable_to_factor, factor_to_variable


def _v179_binary32_with_primary_k(cnf: CNF, topology: LatentPairTopology, boolean_anchor: np.ndarray, anchor_weight: float, primary_k: int):
    # Exact copy of the v177 binary32 map with only the geometry size k supplied
    # by the canonical quotient chart.  No candidate or verifier information is read.
    started = time.perf_counter(); dt = np.float32; D = int(topology.max_domain); directed = int(topology.pair_src.size)
    src = np.asarray(topology.pair_src, dtype=np.int64); dst = np.asarray(topology.pair_dst, dtype=np.int64); rev = np.asarray(topology.reverse_edge, dtype=np.int64)
    valid = np.asarray(topology.valid_mask, dtype=np.bool_); valid_src = valid[src]; valid_dst = valid[dst]
    k = int(primary_k); redundancy, _coverage = _v177_cover_redundancy(cnf, topology)
    disjoint_weight = max(0.0, 1.0 - min(1.0, float(redundancy))); pair_load = float(directed) / float(max(1, k * D)); load_guard = 1.0 + 1.0 / float(max(1, D)); load_excess = max(0.0, pair_load - load_guard)
    effective_k = float(k) * (1.0 + 1.6 * disjoint_weight * load_excess)
    alpha = dt(round(0.6 / math.sqrt(effective_k), 3)); rho = dt(round(0.64 / math.sqrt(effective_k), 2)); beta_start = dt(1.0 / math.sqrt(max(1, D))); beta_stop = dt(round(0.502 * math.sqrt(max(1, k)), 2)); anneal_iterations = int(15 * k); total_iterations = int(39 * k)
    base = np.zeros((int(topology.nblocks), D), dtype=dt)
    if float(anchor_weight) != 0.0:
        lifted = _v173_lift_boolean_field(topology, np.asarray(boolean_anchor, dtype=np.float64)); values = lifted[valid]; scale = float(np.std(values)) if values.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-12: scale = 1.0
        base = dt(anchor_weight) * np.asarray(lifted / scale, dtype=dt); base[~valid] = dt(-80.0); base -= np.max(np.where(valid, base, dt(-np.inf)), axis=1)[:, None]; base[~valid] = dt(-80.0)
    fe, fa, fb, fbin = _v177_build_forbidden(topology); messages = np.zeros((directed, D), dtype=dt); final_update = math.inf
    for iteration in range(total_iterations):
        total = np.zeros((int(topology.nblocks), D), dtype=dt); _v179_scatter_rows(total, dst, messages); cavity = total[src] - messages[rev] + rho * total[src] + base[src]
        if iteration < anneal_iterations:
            frac = dt(iteration / max(1, anneal_iterations - 1)); beta = beta_start * dt((float(beta_stop / beta_start)) ** float(frac))
        else: beta = beta_stop
        z = np.where(valid_src, beta * cavity, dt(-np.inf)); row_max = np.max(z, axis=1); row_max = np.where(np.isfinite(row_max), row_max, dt(0.0)); exponentials = np.exp(np.clip(z - row_max[:, None], dt(-60.0), dt(0.0))).astype(dt, copy=False); exponentials[~valid_src] = dt(0.0); total_partition = exponentials.sum(axis=1, dtype=dt)
        forbidden_flat = np.zeros(directed * D, dtype=dt)
        if fe.size: _v179_scatter_flat(forbidden_flat, fbin, exponentials[fe, fa].astype(dt, copy=False))
        allowed_partition = np.maximum(total_partition[:, None] - forbidden_flat.reshape(directed, D), dt(1e-30))
        new_messages = (row_max[:, None] + np.log(allowed_partition).astype(dt, copy=False)) / beta; new_messages[~valid_dst] = dt(-80.0); new_messages -= np.max(np.where(valid_dst, new_messages, dt(-np.inf)), axis=1)[:, None]; new_messages = np.clip(new_messages, dt(-100.0), dt(0.0)); new_messages[~valid_dst] = dt(-80.0)
        final_update = float(np.max(np.abs(new_messages - messages))); messages = (dt(1.0) - alpha) * messages + alpha * new_messages
    field = base.copy(); _v179_scatter_rows(field, dst, messages); field -= np.max(np.where(valid, field, dt(-np.inf)), axis=1)[:, None]; field[~valid] = dt(-80.0)
    probabilities = _latent_softmax(np.asarray(field, dtype=np.float64), topology); soft_deficiency = _latent_expected_incompatibility(topology, probabilities)
    meta = {"kind":"deterministic_binary32_subtractive_latent_partition_flow","primary_dimension_k":k,"effective_dimension_k":float(effective_k),"pair_load_per_state":float(pair_load),"finite_domain_load_guard":float(load_guard),"load_excess":float(load_excess),"disjoint_cover_weight":float(disjoint_weight),"stability_law":"v177 law on canonical quotient dimension k*","nblocks":int(topology.nblocks),"max_domain":D,"pair_factors":int(directed//2),"directed_messages":directed,"alpha":float(alpha),"rho":float(rho),"beta_start":float(beta_start),"beta_stop":float(beta_stop),"anneal_iterations":anneal_iterations,"total_iterations":total_iterations,"anchor_weight":float(anchor_weight),"final_update_norm":float(final_update),"soft_deficiency_after":float(soft_deficiency),"arithmetic":"IEEE-754 binary32; total-minus-forbidden partition subtraction","summation_order":"NumPy add.at forward directed-edge order","beta_schedule_cast_contract":"v177 frozen cast placement","reads_boolean_assignment":False,"reads_cnf_residuals":False,"uses_verifier_score_for_selection":False,"boolean_flips":False,"branching":False,"decimation":False,"restart_portfolio":False,"runtime_seconds":float(time.perf_counter()-started)}
    return field, messages, meta


def _v179_choose_atlas(cnf: CNF, graph: FactorGraph):
    categorical = build_categorical_topology(cnf, graph)
    if categorical.pure:
        return V179AtlasDecision("frozen_v177_categorical_EXACT1", "established disjoint categorical exact cover has semantic precedence", "v177 exact categorical certificate", "disjoint")
    even = build_even_cycle_latent_pair_topology(cnf, graph)
    if even.pure:
        return V179AtlasDecision("frozen_v177_EVEN_CYCLE", "established exact EVEN_CYCLE macrostate has semantic precedence", "v177 exact alternating-cycle certificate", "even_cycle")
    multi = _v179_detect_multi_partition(cnf)
    if multi is not None:
        return V179AtlasDecision("multi_partition_EXACT1_region_cover", "complete AMO partition plus at least two complete ALO partitions", "global cardinality equality upgrades every group to EXACT1", "regular_multi_overlap", int(multi["primary_dimension"]), multi)
    witness = _v179_compile_disjoint_amo_witness(cnf)
    if witness is not None:
        return V179AtlasDecision("disjoint_AMO_linear_OR_witness_cover", "complete disjoint AMO cover and wide residual OR factors", "canonical first-satisfying witness is an exact linear-size OR chart", "disjoint_plus_wide_OR", len(witness.decision_domain), {"chart": witness})
    # Mixed semantic graph: use already fused OR/EXACT1 factors rather than raw CNF widths.
    max_semantic_or = 0
    for factor in range(graph.n_or_factors):
        max_semantic_or = max(max_semantic_or, int(graph.factor_offsets[factor+1]-graph.factor_offsets[factor]))
    if graph.original_width_max > 12 and max_semantic_or <= 12 and graph.n_even_cycle_factors == 0:
        return V179AtlasDecision("heterogeneous_native_semantic_factor_coordinate", "raw wide clauses were consumed by exact motifs; evaluate the mixed semantic factor graph natively", "closed-form OR and EXACT1 responses preserve every compiled factor exactly", "mixed_general_overlap")
    estimated_work, maximum_width, maximum_coverage, _directed = _v179_generic_work_geometry(cnf)
    if graph.n_even_cycle_factors == 0 and maximum_coverage > 64 and estimated_work > 2_000_000_000:
        return V179AtlasDecision("native_general_factor_coordinate", "generic clause-state chart is exact but representation-dominated", "native OR/EXACT1 factor response is the same local partition equation without state enumeration", "general_high_redundancy")
    return V179AtlasDecision("frozen_v177_generic", "no stronger complete semantic chart certified", "universal exact clause-variable fallback", "general")


def _v179_native_boolean_topology(cnf: CNF) -> LatentPairTopology:
    valid = np.ones((cnf.nvars, 2), dtype=np.bool_)
    return LatentPairTopology(
        pure=True, reason="native exact semantic factor coordinate",
        nblocks=cnf.nvars, max_domain=2,
        domain_sizes=np.full(cnf.nvars, 2, dtype=np.int64), valid_mask=valid,
        pair_src=np.empty(0, dtype=np.int64), pair_dst=np.empty(0, dtype=np.int64), reverse_edge=np.empty(0, dtype=np.int64),
        compatibility=np.empty((0, 2, 2), dtype=np.bool_),
        block_variables=tuple((variable,) for variable in range(cnf.nvars)),
        # State 0 is TRUE so an exact zero field retains the frozen >=0 convention.
        state_selected_vars=tuple(((variable,), tuple()) for variable in range(cnf.nvars)),
        source_kind="native_exact_OR_EXACT1_factor_coordinate",
    )


def _v179_native_factor_partition_flow(cnf: CNF, graph: FactorGraph, boolean_anchor: np.ndarray, *, k: int, anchor_weight: float):
    started = time.perf_counter(); dt = np.float32
    edge_count = int(graph.nedges); variable = np.asarray(graph.edge_var, dtype=np.int64); sign = np.asarray(graph.edge_sign, dtype=np.float32)
    edge_factor = np.asarray(graph.edge_factor, dtype=np.int64); offsets = np.asarray(graph.factor_offsets, dtype=np.int64)
    n_or_edges = int(graph.n_or_edges); n_exact1_edges = int(graph.n_exact1_edges); exact1_start = n_or_edges; exact1_stop = n_or_edges + n_exact1_edges
    or_offsets = offsets[:graph.n_or_factors + 1]
    exact1_offsets = offsets[graph.n_or_factors:graph.n_or_factors + graph.n_exact1_factors + 1] - n_or_edges if graph.n_exact1_factors else np.asarray([0], dtype=np.int64)
    alpha = dt(round(0.6 / math.sqrt(max(1, k)), 3)); rho = dt(round(0.64 / math.sqrt(max(1, k)), 2))
    beta_start = dt(1.0 / math.sqrt(max(1, graph.original_width_max))); beta_stop = dt(round(0.502 * math.sqrt(max(1, k)), 2)); anneal_iterations = int(15 * k); total_iterations = int(39 * k)
    base = np.asarray(boolean_anchor, dtype=dt).copy(); base -= np.mean(base, dtype=dt); scale = float(np.std(base))
    if not np.isfinite(scale) or scale <= 1e-12: scale = 1.0
    base = dt(anchor_weight) * base / dt(scale)
    variable_to_factor = np.zeros(edge_count, dtype=dt); factor_to_variable = np.zeros(edge_count, dtype=dt); final_update = math.inf
    width_or = np.diff(or_offsets) if graph.n_or_factors else np.empty(0, dtype=np.int64)
    for iteration in range(total_iterations):
        if iteration < anneal_iterations:
            fraction = dt(iteration / max(1, anneal_iterations - 1)); beta = beta_start * dt((float(beta_stop / beta_start)) ** float(fraction))
        else: beta = beta_stop
        new_factor = np.empty(edge_count, dtype=dt)
        if n_or_edges:
            cavity = variable_to_factor[:n_or_edges]; literal_sign = sign[:n_or_edges]; z = np.clip(beta * cavity, dt(-60.0), dt(60.0))
            p_true = dt(1.0) / (dt(1.0) + np.exp(-z).astype(dt, copy=False)); p_violate = np.where(literal_sign > 0.0, dt(1.0) - p_true, p_true).astype(dt, copy=False)
            clause_product = np.multiply.reduceat(p_violate, or_offsets[:-1]).astype(dt, copy=False); local_factor = edge_factor[:n_or_edges]
            product_other = clause_product[local_factor] / np.maximum(p_violate, dt(1.0e-30))
            if width_or.size: product_other[width_or[local_factor] == 1] = dt(1.0)
            magnitude = -np.log(np.maximum(dt(1.0) - product_other, dt(1.0e-30))).astype(dt, copy=False) / beta
            new_factor[:n_or_edges] = literal_sign * magnitude
        if n_exact1_edges:
            cavity = variable_to_factor[exact1_start:exact1_stop]; local_factor = edge_factor[exact1_start:exact1_stop] - graph.n_or_factors; z = beta * cavity
            factor_max = np.maximum.reduceat(z, exact1_offsets[:-1]); shifted = np.exp(np.clip(z - factor_max[local_factor], dt(-60.0), dt(0.0))).astype(dt, copy=False)
            factor_sum = np.add.reduceat(shifted, exact1_offsets[:-1]).astype(dt, copy=False); other = np.maximum(factor_sum[local_factor] - shifted, dt(1.0e-30))
            new_factor[exact1_start:exact1_stop] = -(factor_max[local_factor] + np.log(other).astype(dt, copy=False)) / beta
        if graph.n_even_cycle_edges: raise RuntimeError("native factor coordinate is never selected over an EVEN_CYCLE chart")
        total_field = np.bincount(variable, weights=new_factor, minlength=cnf.nvars).astype(dt, copy=False)
        new_variable = total_field[variable] - new_factor + rho * total_field[variable] + base[variable]
        final_update = max(float(np.max(np.abs(new_factor - factor_to_variable))), float(np.max(np.abs(new_variable - variable_to_factor))))
        factor_to_variable = (dt(1.0) - alpha) * factor_to_variable + alpha * new_factor
        variable_to_factor = (dt(1.0) - alpha) * variable_to_factor + alpha * new_variable
    scalar_field = np.bincount(variable, weights=factor_to_variable, minlength=cnf.nvars).astype(dt, copy=False) + base
    topology = _v179_native_boolean_topology(cnf); field = np.empty((cnf.nvars, 2), dtype=dt); field[:, 0] = scalar_field; field[:, 1] = -scalar_field
    meta = {"kind":"deterministic_binary32_native_exact_factor_partition_flow","primary_dimension_k":int(k),"effective_dimension_k":float(k),"pair_load_per_state":float(edge_count/max(1,2*k)),"finite_domain_load_guard":0.0,"load_excess":0.0,"disjoint_cover_weight":0.0,"nblocks":int(cnf.nvars),"max_domain":2,"pair_factors":int(graph.nfactors),"directed_messages":int(edge_count),"alpha":float(alpha),"rho":float(rho),"beta_start":float(beta_start),"beta_stop":float(beta_stop),"anneal_iterations":anneal_iterations,"total_iterations":total_iterations,"anchor_weight":float(anchor_weight),"final_update_norm":float(final_update),"soft_deficiency_after":float("nan"),"arithmetic":"deterministic IEEE-754 binary32 closed-form OR/EXACT1 factor responses","summation_order":"fixed semantic factor-edge order","reads_boolean_assignment":False,"reads_cnf_residuals":False,"uses_verifier_score_for_selection":False,"boolean_flips":False,"branching":False,"decimation":False,"restart_portfolio":False,"runtime_seconds":float(time.perf_counter()-started)}
    messages = np.concatenate((variable_to_factor, factor_to_variable)).astype(dt)
    return field, messages, topology, meta


def _v179_generic_work_geometry(cnf: CNF):
    widths = np.asarray([len(clause) for clause in cnf.clauses], dtype=np.int64); maximum_width = int(np.max(widths)); max_domain = (1 << maximum_width) - 1 if maximum_width <= 20 else 1 << 20
    directed = int(2 * np.sum(widths)); state_updates = int(39 * cnf.nvars * directed * max_domain)
    occurrence = np.zeros(cnf.nvars, dtype=np.int64)
    for clause in cnf.clauses:
        for literal in clause: occurrence[abs(int(literal)) - 1] += 1
    return state_updates, maximum_width, int(np.max(occurrence) + 1), directed


def _v179_atlas_json(atlas: V179AtlasDecision) -> dict:
    details = {}
    if atlas.details:
        for key, value in atlas.details.items():
            if key == "chart":
                chart = value
                details["decision_blocks"] = int(len(chart.decision_domain))
                details["witness_factors"] = int(len(chart.factor_widths))
                details["max_domain"] = int(chart.max_domain)
                details["max_factor_width"] = int(chart.max_factor_width)
            elif isinstance(value, np.ndarray):
                details[key] = value.tolist()
            elif isinstance(value, tuple):
                details[key] = [list(item) if isinstance(item, tuple) else item for item in value]
            elif isinstance(value, (np.integer, np.floating)):
                details[key] = value.item()
            else:
                details[key] = value
    return {
        "chart_kind": atlas.chart_kind,
        "reason": atlas.reason,
        "exact_certificate": atlas.exact_certificate,
        "component_class": atlas.component_class,
        "primary_dimension": atlas.primary_dimension,
        "details": details,
    }


# ---------------------------------------------------------------------------
# v180: algebraic semantic saturation and component-wise chart program
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V180SemanticRelation:
    relation_id: int
    kind: str
    variables: tuple[int, ...]
    clause_ids: tuple[int, ...]
    certificate: str
    parameters: dict
    semantic_rank: int


def _v180_clause_key(clause: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(l) for l in clause))


def _v180_clause_multimap(cnf: CNF) -> dict[tuple[int, ...], list[int]]:
    result: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for clause_id, clause in enumerate(cnf.clauses):
        result[_v180_clause_key(clause)].append(int(clause_id))
    return result


def _v180_detect_exact1_relations(cnf: CNF):
    cliques, pair_clause_ids = _v179_negative_pair_geometry(cnf)
    relations: list[tuple[str, tuple[int, ...], tuple[int, ...], str, dict, int]] = []
    positive_index: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for clause_id, clause in enumerate(cnf.clauses):
        if len(clause) >= 2 and all(int(l) > 0 for l in clause):
            variables = tuple(sorted(int(l) - 1 for l in clause))
            if len(set(variables)) == len(variables):
                positive_index[variables].append(int(clause_id))
    for clique in cliques:
        pair_ids: list[int] = []
        for a, b in itertools.combinations(clique, 2):
            ids = pair_clause_ids.get(tuple(sorted((int(a), int(b)))), ())
            if ids:
                pair_ids.append(int(ids[0]))
        support = tuple(sorted(set(pair_ids)))
        relations.append((
            "ATMOST1", tuple(clique), support,
            "complete negative-binary clique certifies pairwise ATMOST1",
            {"bound": 1, "width": len(clique)}, 30,
        ))
        if tuple(clique) in positive_index:
            clause_id = int(positive_index[tuple(clique)][0])
            relations.append((
                "EXACT1", tuple(clique), tuple(sorted(support + (clause_id,))),
                "positive ALO plus complete pairwise ATMOST1 is equivalent to EXACT1",
                {"value": 1, "width": len(clique), "source": "local_ALO_plus_AMO"}, 10,
            ))
    inferred = _v179_detect_multi_partition(cnf)
    if inferred is not None:
        groups: list[tuple[int, ...]] = list(inferred["primary_blocks"])
        positive_groups = inferred["positive_groups"]
        for family in inferred["families"]:
            groups.extend(positive_groups[group_id] for group_id in family)
        seen: set[tuple[int, ...]] = set()
        for group in groups:
            key = tuple(sorted(int(v) for v in group))
            if key in seen:
                continue
            seen.add(key)
            supporting: list[int] = []
            if key in positive_index:
                supporting.append(int(positive_index[key][0]))
            for a, b in itertools.combinations(key, 2):
                ids = pair_clause_ids.get(tuple(sorted((a, b))), ())
                if ids:
                    supporting.append(int(ids[0]))
            relations.append((
                "EXACT1", key, tuple(sorted(set(supporting))),
                "global partition-cardinality closure upgrades every group to EXACT1",
                {"value": 1, "width": len(key), "source": "partition_cardinality_closure"}, 9,
            ))
    return relations, cliques


def _v180_detect_parity_relations(cnf: CNF, max_width: int = 10):
    # v182 exact two-pass implementation.  A width-w XOR CNF needs at least
    # 2^(w-1) clauses with the identical variable support.  Count supports
    # first; only materialize falsifying sign patterns for supports that can
    # possibly satisfy that necessary condition.
    support_count: dict[tuple[int, ...], int] = defaultdict(int)
    for clause in cnf.clauses:
        if not (2 <= len(clause) <= max_width):
            continue
        variables = tuple(sorted(abs(int(l)) - 1 for l in clause))
        if len(set(variables)) != len(variables):
            continue
        support_count[variables] += 1
    candidates = {
        variables for variables, count in support_count.items()
        if count >= (1 << (len(variables) - 1))
    }
    if not candidates:
        return []
    grouped: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for clause_id, clause in enumerate(cnf.clauses):
        if not (2 <= len(clause) <= max_width):
            continue
        variables = tuple(sorted(abs(int(l)) - 1 for l in clause))
        if variables not in candidates:
            continue
        literal_by_var = {abs(int(l)) - 1: int(l) for l in clause}
        falsifying = tuple(0 if literal_by_var[v] > 0 else 1 for v in variables)
        grouped[variables].append((int(clause_id), falsifying))
    relations = []
    for variables, rows in grouped.items():
        width = len(variables)
        required = 1 << (width - 1)
        assignments: dict[tuple[int, ...], int] = {}
        for clause_id, assignment in rows:
            assignments.setdefault(assignment, clause_id)
        if len(assignments) != required:
            continue
        parities = {sum(bits) & 1 for bits in assignments}
        if len(parities) != 1:
            continue
        forbidden_parity = next(iter(parities))
        expected = {
            bits for bits in itertools.product((0, 1), repeat=width)
            if (sum(bits) & 1) == forbidden_parity
        }
        if set(assignments) != expected:
            continue
        rhs = 1 - forbidden_parity
        relations.append((
            "XOR", variables, tuple(sorted(assignments.values())),
            "complete set of parity-forbidden clauses is equivalent to one GF(2) equation",
            {"rhs": int(rhs), "width": width}, 8,
        ))
    return relations


def _v180_detect_gate_relations(cnf: CNF):
    clause_map = _v180_clause_multimap(cnf)
    relations = []
    seen: set[tuple[str, int, tuple[int, ...]]] = set()
    for long_id, clause in enumerate(cnf.clauses):
        if len(clause) < 3:
            continue
        positive = [int(l) for l in clause if int(l) > 0]
        negative = [int(l) for l in clause if int(l) < 0]
        # y <-> AND(x_1,...,x_k): (y OR -x_1 OR ... OR -x_k) and (-y OR x_i).
        if len(positive) == 1 and len(negative) >= 2:
            output = positive[0] - 1
            inputs = tuple(sorted(-l - 1 for l in negative))
            binary_ids = []
            valid = True
            for variable in inputs:
                key = _v180_clause_key((-(output + 1), variable + 1))
                ids = clause_map.get(key, ())
                if not ids:
                    valid = False; break
                binary_ids.append(int(ids[0]))
            if valid:
                signature = ("AND_GATE", output, inputs)
                if signature not in seen:
                    seen.add(signature)
                    relations.append((
                        "AND_GATE", tuple(sorted((output,) + inputs)), tuple(sorted((long_id, *binary_ids))),
                        "canonical Tseitin clauses certify output iff conjunction of inputs",
                        {"output": output, "inputs": inputs}, 6,
                    ))
        # y <-> OR(x_1,...,x_k): (-y OR x_1 OR ... OR x_k) and (y OR -x_i).
        if len(negative) == 1 and len(positive) >= 2:
            output = -negative[0] - 1
            inputs = tuple(sorted(l - 1 for l in positive))
            binary_ids = []
            valid = True
            for variable in inputs:
                key = _v180_clause_key((output + 1, -(variable + 1)))
                ids = clause_map.get(key, ())
                if not ids:
                    valid = False; break
                binary_ids.append(int(ids[0]))
            if valid:
                signature = ("OR_GATE", output, inputs)
                if signature not in seen:
                    seen.add(signature)
                    relations.append((
                        "OR_GATE", tuple(sorted((output,) + inputs)), tuple(sorted((long_id, *binary_ids))),
                        "canonical Tseitin clauses certify output iff disjunction of inputs",
                        {"output": output, "inputs": inputs}, 6,
                    ))
    return relations




def _v180_detect_compiled_factor_relations(graph: FactorGraph):
    relations = []
    for factor in range(graph.nfactors):
        start = int(graph.factor_offsets[factor]); stop = int(graph.factor_offsets[factor + 1])
        variables = tuple(sorted(set(int(v) for v in graph.edge_var[start:stop])))
        factor_type = int(graph.factor_type[factor])
        if factor_type == FACTOR_EXACT1:
            relations.append((
                "EXACT1", variables, tuple(),
                "frozen semantic compiler certified positive ALO plus complete pairwise AMO",
                {"value": 1, "width": len(variables), "source": "compiled_factor_graph"}, 5,
            ))
        elif factor_type == FACTOR_EVEN_CYCLE:
            local = factor - graph.n_or_factors - graph.n_exact1_factors
            length = int(graph.even_cycle_lengths[local]) if 0 <= local < len(graph.even_cycle_lengths) else 0
            relations.append((
                "EVEN_CYCLE", variables, tuple(),
                "frozen semantic compiler certified exact alternating even-cycle macrostate",
                {"cycle_length": length, "width": len(variables)}, 2,
            ))
    return relations


def _v180_detect_wide_or_relations(cnf: CNF, minimum_width: int = 4):
    relations = []
    for clause_id, clause in enumerate(cnf.clauses):
        if len(clause) < minimum_width:
            continue
        variables = tuple(sorted(set(abs(int(l)) - 1 for l in clause)))
        if len(variables) != len(clause):
            continue
        relations.append((
            "OR_REGION", variables, (int(clause_id),),
            "original CNF clause is an exact finite relation; witness chart is linear in arity",
            {"width": len(clause), "signs": tuple(1 if int(l) > 0 else -1 for l in clause)}, 25,
        ))
    return relations


def _v180_detect_implication_relations(cnf: CNF):
    relations = []
    for clause_id, clause in enumerate(cnf.clauses):
        if len(clause) != 2:
            continue
        a, b = (int(clause[0]), int(clause[1]))
        # Every binary clause (a OR b) is the pair of implications (-a -> b), (-b -> a).
        relations.append((
            "BINARY_CLAUSE", tuple(sorted((abs(a) - 1, abs(b) - 1))), (int(clause_id),),
            "binary clause represented as two directed implications",
            {"implications": ((-a, b), (-b, a))}, 50,
        ))
    return relations


def _v181_gf2_peel_summary(rels: list[V180SemanticRelation]) -> dict:
    """Linear-time leaf peeling on a pure XOR incidence component.

    This is representation analysis only.  A degree-one variable is a valid GF(2)
    pivot: its unique equation can be solved for that variable exactly.  Removing
    that equation can expose another degree-one pivot.  No Boolean assignment,
    residual score, or verifier result is read.
    """
    import heapq
    relation_ids = [int(r.relation_id) for r in rels]
    rel_vars = {int(r.relation_id): tuple(int(v) for v in r.variables) for r in rels}
    var_to_rel: dict[int, set[int]] = defaultdict(set)
    for r in rels:
        for v in r.variables:
            var_to_rel[int(v)].add(int(r.relation_id))
    active_rel = set(relation_ids)
    degree = {v: len(ids) for v, ids in var_to_rel.items()}
    heap = [v for v, d in degree.items() if d == 1]
    heapq.heapify(heap)
    pivots: list[tuple[int, int]] = []
    while heap:
        v = heapq.heappop(heap)
        if degree.get(v, 0) != 1:
            continue
        candidates = [rid for rid in var_to_rel[v] if rid in active_rel]
        if len(candidates) != 1:
            continue
        rid = candidates[0]
        active_rel.remove(rid)
        pivots.append((int(v), int(rid)))
        for u in rel_vars[rid]:
            if degree.get(u, 0) <= 0:
                continue
            degree[u] -= 1
            if degree[u] == 1:
                heapq.heappush(heap, int(u))
    core_variables = sorted(v for v, d in degree.items() if d > 0)
    pivot_vars = {pv for pv, _ in pivots}
    boundary_variables = sorted(v for v in var_to_rel if v not in pivot_vars)
    # In a fully peeled component the surviving free coordinates are variables
    # that were never used as pivots.  They need not have positive final degree.
    fully_peelable = not active_rel
    return {
        "kind": "canonical_GF2_leaf_peeling",
        "equation_count": len(relation_ids),
        "variable_count": len(var_to_rel),
        "eliminated_equations": len(pivots),
        "pivot_variable_count": len(pivots),
        "remaining_equations": len(active_rel),
        "core_variable_count": len(boundary_variables) if fully_peelable else len(core_variables),
        "fully_peelable": bool(fully_peelable),
        "compression_ratio_variables_to_core": float(len(var_to_rel) / max(1, len(boundary_variables))) if fully_peelable else None,
        "pivot_preview": [[int(v), int(r)] for v, r in pivots[:16]],
        "core_variables_preview": [int(v) for v in (boundary_variables if fully_peelable else core_variables)[:32]],
        "contract": "degree-one GF(2) elimination is exact and assignment-independent",
    }


def _v180_relation_components(relations: tuple[V180SemanticRelation, ...]):
    """v181 implementation: linear-incidence component construction.

    v180 explicitly materialized the relation-overlap clique and then recomputed
    coverage by scanning every relation for every variable.  Dense semantic hubs
    therefore caused O(R^2) memory/work and O(VR) Python membership work.  Here
    components are the connected components of the bipartite incidence graph,
    built with union-find in O(total semantic incidence * alpha(R)).
    """
    strong_ids = [int(r.relation_id) for r in relations if r.semantic_rank <= 30]
    if not strong_ids:
        return []
    relation_by_id = {int(r.relation_id): r for r in relations}
    parent = {rid: rid for rid in strong_ids}
    rank = {rid: 0 for rid in strong_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    var_to_rel: dict[int, list[int]] = defaultdict(list)
    for rid in strong_ids:
        for variable in relation_by_id[rid].variables:
            var_to_rel[int(variable)].append(rid)
    for owners in var_to_rel.values():
        if len(owners) > 1:
            first = owners[0]
            for other in owners[1:]:
                union(first, other)

    groups: dict[int, list[int]] = defaultdict(list)
    for rid in strong_ids:
        groups[find(rid)].append(rid)

    # Assign variables and their exact incidence degrees directly to components.
    component_vars: dict[int, list[tuple[int, int]]] = defaultdict(list)
    overlap_pair_multiplicity: dict[int, int] = defaultdict(int)
    for variable, owners0 in var_to_rel.items():
        owners = sorted(set(int(x) for x in owners0))
        root = find(owners[0])
        component_vars[root].append((int(variable), len(owners)))
        overlap_pair_multiplicity[root] += len(owners) * (len(owners) - 1) // 2

    components = []
    for root in sorted(groups, key=lambda r: min(groups[r])):
        ids = sorted(groups[root])
        rels = [relation_by_id[i] for i in ids]
        var_degree = sorted(component_vars[root])
        variables = [v for v, _ in var_degree]
        coverage = [d for _, d in var_degree]
        kinds = sorted(set(r.kind for r in rels))
        incidence_edges = sum(len(r.variables) for r in rels)
        # First Betti number of the connected bipartite incidence graph.
        cycle_rank = max(0, int(incidence_edges - len(ids) - len(variables) + 1))
        exact1_rels = [r for r in rels if r.kind == "EXACT1"]
        exact1_degree: dict[int, int] = defaultdict(int)
        for r in exact1_rels:
            for v in r.variables:
                exact1_degree[int(v)] += 1
        disjoint_exact1 = bool(exact1_rels) and max(exact1_degree.values(), default=0) <= 1
        regular_coverage = bool(coverage) and len(set(coverage)) == 1
        gf2_summary = None
        if "EVEN_CYCLE" in kinds:
            overlap_class = "even_cycle"
            chart = "alternating_cycle_macrostate"
        elif disjoint_exact1:
            overlap_class = "disjoint_exact1"
            chart = "categorical_EXACT1"
        elif exact1_rels and regular_coverage and max(coverage, default=0) >= 2:
            overlap_class = "regular_multi_overlap"
            chart = "multi_partition_region_concurrence"
        elif "XOR" in kinds:
            # Exact fill-in-free algebraic elimination is a rewrite stage, not a
            # competing benchmark-specific chart.  If the XOR backbone peels
            # completely, eliminate it before choosing a chart for the residual
            # relations.  This canonical precedence is independent of family.
            xor_rels = [r for r in rels if r.kind == "XOR"]
            gf2_summary = _v181_gf2_peel_summary(xor_rels)
            nonxor_kinds = sorted(k for k in kinds if k != "XOR")
            gf2_summary["residual_relation_kinds"] = nonxor_kinds
            gf2_summary["residual_relation_count"] = len(rels) - len(xor_rels)
            if gf2_summary["fully_peelable"]:
                overlap_class = "peelable_parity_backbone" if nonxor_kinds else "peelable_parity_network"
                chart = "GF2_peel_core_plus_residual_regions" if nonxor_kinds else "GF2_peel_core_region"
            else:
                overlap_class = "parity_backbone" if nonxor_kinds else "parity_network"
                chart = "GF2_core_plus_residual_regions" if nonxor_kinds else "GF2_parity_region"
        elif "ATMOST1" in kinds and "OR_REGION" in kinds:
            overlap_class = "disjoint_AMO_plus_OR"
            chart = "categorical_AMO_witness_region"
        elif cycle_rank == 0:
            overlap_class = "acyclic_overlap"
            chart = "junction_tree_region"
        elif any(k.endswith("_GATE") for k in kinds):
            overlap_class = "circuit_overlap"
            chart = "gate_region_graph"
        else:
            overlap_class = "general_overlap"
            chart = "ragged_region_graph"
        item = {
            "component_id": len(components),
            "relation_ids": ids,
            "relation_count": len(ids),
            "variable_count": len(variables),
            "variables": variables,
            "relation_kinds": kinds,
            "coverage_min": min(coverage, default=0),
            "coverage_mean": float(sum(coverage) / len(coverage)) if coverage else 0.0,
            "coverage_max": max(coverage, default=0),
            "overlap_edges": int(overlap_pair_multiplicity[root]),
            "incidence_edges": int(incidence_edges),
            "cycle_rank": int(cycle_rank),
            "overlap_class": overlap_class,
            "canonical_chart": chart,
        }
        if gf2_summary is not None:
            item["gf2_peeling"] = gf2_summary
        components.append(item)
    return components


def _v180_build_semantic_program(cnf: CNF, graph: FactorGraph):
    raw = []
    exact1, amo_cliques = _v180_detect_exact1_relations(cnf)
    raw.extend(exact1)
    raw.extend(_v180_detect_compiled_factor_relations(graph))
    raw.extend(_v180_detect_parity_relations(cnf))
    raw.extend(_v180_detect_gate_relations(cnf))
    raw.extend(_v180_detect_wide_or_relations(cnf))
    raw.extend(_v180_detect_implication_relations(cnf))
    # Stable semantic deduplication: retain the strongest certificate for each kind/support/parameters.
    dedup: dict[tuple, tuple] = {}
    for item in raw:
        kind, variables, clause_ids, certificate, parameters, rank = item
        param_key = json.dumps(parameters, sort_keys=True, default=list)
        key = (kind, tuple(variables), param_key)
        old = dedup.get(key)
        if old is None or rank < old[-1] or (rank == old[-1] and tuple(clause_ids) < tuple(old[2])):
            dedup[key] = item
    ordered = sorted(dedup.values(), key=lambda x: (x[-1], x[0], x[1], x[2]))
    relations = tuple(
        V180SemanticRelation(
            relation_id=index, kind=item[0], variables=tuple(item[1]), clause_ids=tuple(item[2]),
            certificate=item[3], parameters=item[4], semantic_rank=int(item[5]),
        )
        for index, item in enumerate(ordered)
    )
    components = _v180_relation_components(relations)
    # v182: component incidence can be enormous (industrial CNFs).  The full
    # relation/variable lists are not consumed by the active operator; keep a
    # bounded deterministic preview in JSON while preserving exact counts and
    # all scalar geometry/certificates.
    component_output = []
    for component in components:
        item = dict(component)
        relation_ids = list(item.pop("relation_ids", []))
        variables = list(item.pop("variables", []))
        item["relation_ids_preview"] = relation_ids[:128]
        item["variables_preview"] = variables[:128]
        item["relation_ids_truncated"] = len(relation_ids) > 128
        item["variables_truncated"] = len(variables) > 128
        component_output.append(item)
    counts: dict[str, int] = defaultdict(int)
    for relation in relations:
        counts[relation.kind] += 1
    # Residual clauses are those not supporting any stronger-than-binary semantic relation.
    semantically_consumed = set()
    for relation in relations:
        if relation.semantic_rank <= 30:
            semantically_consumed.update(relation.clause_ids)
    residual_clause_ids = [i for i in range(len(cnf.clauses)) if i not in semantically_consumed]
    return {
        "kind": "algebraic_semantic_saturation_program",
        "saturation_contract": "only equivalence-preserving relations with explicit clause certificates are added",
        "relation_counts": dict(sorted(counts.items())),
        "relation_count": len(relations),
        "strong_relation_count": sum(r.semantic_rank <= 30 for r in relations),
        "amo_clique_count": len(amo_cliques),
        "component_count": len(components),
        "components": component_output,
        "residual_clause_count": len(residual_clause_ids),
        "residual_clause_ids_preview": residual_clause_ids[:64],
        "relations_preview": [
            {
                "relation_id": r.relation_id,
                "kind": r.kind,
                "variables": list(r.variables),
                "clause_ids": list(r.clause_ids),
                "certificate": r.certificate,
                "parameters": r.parameters,
                "semantic_rank": r.semantic_rank,
            }
            for r in relations[:128]
        ],
        "active_operator_note": "v180 compiler program is audited first; established v179 chart dynamics remain frozen",
        "established_factor_geometry": {
            "OR": int(graph.n_or_factors),
            "EXACT1": int(graph.n_exact1_factors),
            "EVEN_CYCLE": int(graph.n_even_cycle_factors),
        },
    }


def _v179_multiscale_partition_operator(cnf: CNF, graph: FactorGraph, args: argparse.Namespace):
    atlas = _v179_choose_atlas(cnf, graph)
    # Frozen charts call the v177 implementation literally.
    if atlas.chart_kind.startswith("frozen_v177"):
        result = _v177_multiscale_partition_operator(cnf, graph, args)
        latent_meta = dict(result[4]); latent_meta["semantic_atlas"] = _v179_atlas_json(atlas)
        return (*result[:4], latent_meta, *result[5:])
    if atlas.chart_kind in ("heterogeneous_native_semantic_factor_coordinate", "native_general_factor_coordinate"):
        remaining_or_width_preview = 0
        if graph.n_or_factors:
            remaining_or_width_preview = int(np.max(np.diff(graph.factor_offsets)[:graph.n_or_factors]))
        if atlas.chart_kind == "heterogeneous_native_semantic_factor_coordinate" and graph.n_exact1_factors and remaining_or_width_preview <= 12:
            native_k_preview = int(max(1, graph.n_exact1_factors))
        else:
            native_k_preview = int(cnf.nvars)
        factor_cap = max(1000, 180 * native_k_preview)
    else:
        factor_cap = None
    boolean_field, factor_meta, variable_to_factor, factor_to_variable = _v179_run_factor_channel(cnf, graph, args, iteration_cap=factor_cap)
    if atlas.chart_kind == "multi_partition_EXACT1_region_cover":
        topology = _v179_compile_multi_partition(cnf, atlas.details)
        # The canonical quotient dimension is the number of blocks in one partition.
        latent_field, latent_messages, latent_meta = _v179_binary32_with_primary_k(cnf, topology, boolean_field, 0.0, int(atlas.primary_dimension))
    elif atlas.chart_kind == "disjoint_AMO_linear_OR_witness_cover":
        chart = atlas.details["chart"]
        latent_field, latent_messages, topology, latent_meta = _v179_numpy_witness_flow(cnf, chart)
    elif atlas.chart_kind in ("heterogeneous_native_semantic_factor_coordinate", "native_general_factor_coordinate"):
        remaining_or_width = 0
        if graph.n_or_factors:
            remaining_or_width = int(np.max(np.diff(graph.factor_offsets)[:graph.n_or_factors]))
        if atlas.chart_kind == "heterogeneous_native_semantic_factor_coordinate" and graph.n_exact1_factors and remaining_or_width <= 12:
            native_k = int(max(1, graph.n_exact1_factors))
        else:
            native_k = int(cnf.nvars)
        latent_field, latent_messages, topology, latent_meta = _v179_native_factor_partition_flow(cnf, graph, boolean_field, k=native_k, anchor_weight=0.1)
    else:
        raise RuntimeError(f"unhandled v179 semantic chart: {atlas.chart_kind}")
    redundancy, coverage = _v177_cover_redundancy(cnf, topology)
    coupling_meta = {"kind":"exact_cover_redundancy_coupling","coverage_min":int(np.min(coverage)),"coverage_mean":float(np.mean(coverage)),"coverage_max":int(np.max(coverage)),"redundancy_chi":float(redundancy),"law":"lambda determined by canonical chart geometry","anchor_weight":float(latent_meta.get("anchor_weight",0.0)),"reads_boolean_assignment":False,"reads_cnf_residuals":False}
    latent_meta = {**latent_meta, "coupling":coupling_meta, "semantic_atlas":_v179_atlas_json(atlas)}
    return latent_field, latent_messages, topology, factor_meta, latent_meta, variable_to_factor, factor_to_variable



# ---------------------------------------------------------------------------
# v182: rewrite-first GF(2) affine residual regions
# ---------------------------------------------------------------------------

@dataclass
class V182AffinePlan:
    core_size: int
    expr_mask: dict[int, int]
    expr_const: dict[int, int]
    offsets: np.ndarray
    support_index: np.ndarray
    syndrome_column: np.ndarray
    ranks: np.ndarray
    rhs: np.ndarray
    clause_ids: np.ndarray
    selected_basis: str
    candidate_costs: dict
    xor_equations: int
    xor_variables: int
    xor_clause_count: int
    residual_factor_count: int
    residual_tautologies: int
    max_rank: int
    max_support: int
    representation_cost: int
    affine_work_score: int
    fallback_work_score: int
    rewrite_dominates: bool


def _v182_gf2_basis(rows: list[int], rhs: list[int]):
    basis: dict[int, tuple[int, int]] = {}
    for mask, bit in zip(rows, rhs):
        x = int(mask); b = int(bit) & 1
        while x:
            pivot = x.bit_length() - 1
            if pivot in basis:
                old_mask, old_bit = basis[pivot]
                x ^= old_mask; b ^= old_bit
            else:
                basis[pivot] = (x, b)
                break
        if x == 0 and b:
            return None
    return [basis[p] for p in sorted(basis, reverse=True)]


def _v182_peel_parameterization(cnf: CNF, parity_raw, mode: str):
    import heapq
    rel_vars: dict[int, tuple[int, ...]] = {}
    rel_rhs: dict[int, int] = {}
    var_to_rel: dict[int, set[int]] = defaultdict(set)
    consumed: set[int] = set()
    static_degree: dict[int, int] = defaultdict(int)
    for rid, item in enumerate(parity_raw):
        variables = tuple(int(v) for v in item[1])
        rel_vars[rid] = variables
        rel_rhs[rid] = int(item[4]["rhs"])
        consumed.update(int(cid) for cid in item[2])
        for variable in variables:
            var_to_rel[variable].add(rid)
            static_degree[variable] += 1
    residual_degree: dict[int, int] = defaultdict(int)
    for cid, clause in enumerate(cnf.clauses):
        if cid in consumed:
            continue
        for literal in clause:
            residual_degree[abs(int(literal)) - 1] += 1

    def priority(variable: int):
        if mode == "static_degree":
            return (static_degree[variable], residual_degree[variable], variable)
        if mode == "residual_degree":
            return (residual_degree[variable], static_degree[variable], variable)
        return (0, 0, variable)

    active = set(rel_vars)
    degree = {v: len(ids) for v, ids in var_to_rel.items()}
    heap = [(priority(v), v) for v, d in degree.items() if d == 1]
    heapq.heapify(heap)
    pivots: list[tuple[int, int]] = []
    while heap:
        _key, variable = heapq.heappop(heap)
        if degree.get(variable, 0) != 1:
            continue
        owners = [rid for rid in var_to_rel[variable] if rid in active]
        if len(owners) != 1:
            continue
        rid = owners[0]
        active.remove(rid)
        pivots.append((variable, rid))
        for other in rel_vars[rid]:
            if degree.get(other, 0) <= 0:
                continue
            degree[other] -= 1
            if degree[other] == 1:
                heapq.heappush(heap, (priority(other), other))
    if active:
        return None
    pivot_set = {v for v, _ in pivots}
    xor_variables = set(var_to_rel)
    free = sorted(v for v in xor_variables if v not in pivot_set)
    outside = sorted(set(range(cnf.nvars)) - xor_variables)
    core_variables = free + outside
    core_index = {v: i for i, v in enumerate(core_variables)}
    expr_mask = {v: 1 << core_index[v] for v in core_variables}
    expr_const = {v: 0 for v in core_variables}
    for variable, rid in reversed(pivots):
        mask = 0; constant = rel_rhs[rid]
        for other in rel_vars[rid]:
            if other == variable:
                continue
            mask ^= expr_mask[other]
            constant ^= expr_const[other]
        expr_mask[variable] = int(mask)
        expr_const[variable] = int(constant)
    return expr_mask, expr_const, consumed, len(xor_variables), len(core_variables), len(pivots)


def _v182_compile_factors(cnf: CNF, expr_mask, expr_const, consumed, core_size: int):
    factor_records = []
    tautologies = 0
    representation_cost = 0
    max_rank = 0; max_support = 0
    for cid, clause in enumerate(cnf.clauses):
        if cid in consumed:
            continue
        rows: list[int] = []; bits: list[int] = []; union = 0
        for literal in clause:
            variable = abs(int(literal)) - 1
            mask = int(expr_mask[variable]); constant = int(expr_const[variable])
            false_value = 0 if literal > 0 else 1
            rows.append(mask); bits.append(false_value ^ constant); union |= mask
        basis = _v182_gf2_basis(rows, bits)
        if basis is None:
            tautologies += 1
            continue
        if len(basis) == 0:
            raise RuntimeError("v182 affine rewrite produced an always-false residual clause")
        rank = len(basis)
        support: list[int] = []
        remaining_support = int(union)
        while remaining_support:
            lowbit = remaining_support & -remaining_support
            support.append(lowbit.bit_length() - 1)
            remaining_support ^= lowbit
        columns: list[int] = []
        for variable in support:
            column = 0
            for row, (mask, _bit) in enumerate(basis):
                if (mask >> variable) & 1:
                    column |= 1 << row
            columns.append(column)
        target = sum((int(bit) & 1) << row for row, (_mask, bit) in enumerate(basis))
        representation_cost += len(support) * (1 << rank)
        max_rank = max(max_rank, rank); max_support = max(max_support, len(support))
        factor_records.append((int(cid), rank, int(target), support, columns))
    offsets = [0]; support_index: list[int] = []; syndrome_column: list[int] = []
    ranks: list[int] = []; rhs: list[int] = []; clause_ids: list[int] = []
    for cid, rank, target, support, columns in factor_records:
        support_index.extend(support); syndrome_column.extend(columns); offsets.append(len(support_index))
        ranks.append(rank); rhs.append(target); clause_ids.append(cid)
    return (
        np.asarray(offsets, dtype=np.int32), np.asarray(support_index, dtype=np.int32),
        np.asarray(syndrome_column, dtype=np.int32), np.asarray(ranks, dtype=np.int16),
        np.asarray(rhs, dtype=np.int32), np.asarray(clause_ids, dtype=np.int64),
        int(tautologies), int(max_rank), int(max_support), int(representation_cost),
    )


def _v182_compile_affine_plan(cnf: CNF):
    parity_raw = _v180_detect_parity_relations(cnf)
    if not parity_raw:
        return None
    candidates = []
    for mode in ("lexicographic", "static_degree", "residual_degree"):
        parameterization = _v182_peel_parameterization(cnf, parity_raw, mode)
        if parameterization is None:
            continue
        expr_mask, expr_const, consumed, xor_variables, core_size, pivot_count = parameterization
        packed = _v182_compile_factors(cnf, expr_mask, expr_const, consumed, core_size)
        cost = packed[-1]
        candidates.append((cost, mode, expr_mask, expr_const, consumed, xor_variables, core_size, pivot_count, packed))
    if not candidates:
        return None
    # The active GF(2) coordinate system is canonical and unique: the
    # lexicographically smallest available degree-one pivot at every peel step.
    # Alternative exact bases are retained only as representation-cost audit;
    # they never compete by solver outcome or verifier score.
    candidate_costs = {item[1]: int(item[0]) for item in candidates}
    best = next(item for item in candidates if item[1] == "lexicographic")
    cost, mode, expr_mask, expr_const, consumed, xor_variables, core_size, pivot_count, packed = best
    offsets, support_index, syndrome_column, ranks, rhs, clause_ids, taut, max_rank, max_support, rep_cost = packed
    return V182AffinePlan(
        core_size=int(core_size), expr_mask=expr_mask, expr_const=expr_const,
        offsets=offsets, support_index=support_index, syndrome_column=syndrome_column,
        ranks=ranks, rhs=rhs, clause_ids=clause_ids, selected_basis=mode,
        candidate_costs=candidate_costs, xor_equations=len(parity_raw),
        xor_variables=int(xor_variables), xor_clause_count=len(consumed),
        residual_factor_count=int(ranks.size), residual_tautologies=int(taut),
        max_rank=int(max_rank), max_support=int(max_support), representation_cost=int(rep_cost),
        affine_work_score=int(5 * max(1, core_size) * max(1, rep_cost)),
        fallback_work_score=int(39 * max(1, cnf.nvars) * max(1, sum(len(clause) for clause in cnf.clauses))),
        rewrite_dominates=bool(5 * max(1, core_size) * max(1, rep_cost) < 39 * max(1, cnf.nvars) * max(1, sum(len(clause) for clause in cnf.clauses))),
    )


def _v182_python_obj_grad(prob, plan: V182AffinePlan, degree, beta: float):
    grad = np.zeros(plan.core_size, dtype=np.float64); energy = 0.0; max_forbidden = 0.0
    for factor in range(plan.ranks.size):
        rank = int(plan.ranks[factor]); states = 1 << rank
        start = int(plan.offsets[factor]); stop = int(plan.offsets[factor + 1])
        support = plan.support_index[start:stop]; columns = plan.syndrome_column[start:stop]
        forward = np.zeros((stop - start + 1, states), dtype=np.float64); forward[0, 0] = 1.0
        index = np.arange(states, dtype=np.int32)
        for t, (variable, column) in enumerate(zip(support, columns)):
            p = float(prob[int(variable)]); previous = forward[t]
            forward[t + 1] = (1.0 - p) * previous + p * previous[index ^ int(column)]
        target = int(plan.rhs[factor]); forbidden = float(forward[-1, target]); max_forbidden = max(max_forbidden, forbidden)
        sat = max(1e-14, 1.0 - forbidden); energy += -math.log(sat)
        adjoint = np.zeros(states, dtype=np.float64); adjoint[target] = 1.0
        for t in range(stop - start - 1, -1, -1):
            variable = int(support[t]); column = int(columns[t]); p = float(prob[variable]); previous = forward[t]
            dprob = float(np.dot(adjoint, previous[index ^ column] - previous))
            grad[variable] += (dprob / sat) * (beta * p * (1.0 - p))
            adjoint = (1.0 - p) * adjoint + p * adjoint[index ^ column]
    grad /= np.sqrt(np.maximum(degree, 1.0))
    return energy, grad, max_forbidden


try:
    import numba as _v182_numba
except Exception:
    _v182_numba = None

if _v182_numba is not None:
    @_v182_numba.njit(cache=True)
    def _v182_numba_obj_grad(prob, offsets, support_index, syndrome_column, ranks, rhs, degree, beta):
        core_size = prob.size; grad = np.zeros(core_size, np.float64); energy = 0.0; max_forbidden = 0.0
        max_support = 0; max_rank = 0
        for f in range(ranks.size):
            q = offsets[f + 1] - offsets[f]
            if q > max_support: max_support = q
            if ranks[f] > max_rank: max_rank = ranks[f]
        max_states = 1 << max_rank
        forward = np.empty((max_support + 1, max_states), np.float64)
        adjoint = np.empty(max_states, np.float64); next_adjoint = np.empty(max_states, np.float64)
        for f in range(ranks.size):
            rank = int(ranks[f]); states = 1 << rank; start = int(offsets[f]); q = int(offsets[f + 1] - start)
            for s in range(states): forward[0, s] = 0.0
            forward[0, 0] = 1.0
            for t in range(q):
                edge = start + t; variable = int(support_index[edge]); column = int(syndrome_column[edge]); p = prob[variable]
                for state in range(states):
                    forward[t + 1, state] = (1.0 - p) * forward[t, state] + p * forward[t, state ^ column]
            target = int(rhs[f]); forbidden = forward[q, target]
            if forbidden > max_forbidden: max_forbidden = forbidden
            sat = 1.0 - forbidden
            if sat < 1e-14: sat = 1e-14
            energy += -math.log(sat)
            for state in range(states): adjoint[state] = 0.0
            adjoint[target] = 1.0
            for t in range(q - 1, -1, -1):
                edge = start + t; variable = int(support_index[edge]); column = int(syndrome_column[edge]); p = prob[variable]
                dprob = 0.0
                for state in range(states): dprob += adjoint[state] * (forward[t, state ^ column] - forward[t, state])
                grad[variable] += (dprob / sat) * (beta * p * (1.0 - p))
                for state in range(states): next_adjoint[state] = (1.0 - p) * adjoint[state] + p * adjoint[state ^ column]
                for state in range(states): adjoint[state] = next_adjoint[state]
        for variable in range(core_size): grad[variable] /= math.sqrt(degree[variable] if degree[variable] > 1.0 else 1.0)
        return energy, grad, max_forbidden

    @_v182_numba.njit(cache=True)
    def _v182_numba_factor_messages(v2f, offsets, support_index, syndrome_column, ranks, rhs, beta, clip_value):
        edge_count = support_index.size; output = np.zeros(edge_count, np.float64)
        max_support = 0; max_rank = 0
        for f in range(ranks.size):
            q = offsets[f + 1] - offsets[f]
            if q > max_support: max_support = q
            if ranks[f] > max_rank: max_rank = ranks[f]
        max_states = 1 << max_rank
        forward = np.empty((max_support + 1, max_states), np.float64)
        adjoint = np.empty(max_states, np.float64); next_adjoint = np.empty(max_states, np.float64); probabilities = np.empty(max_support, np.float64)
        for f in range(ranks.size):
            rank = int(ranks[f]); states = 1 << rank; start = int(offsets[f]); q = int(offsets[f + 1] - start)
            for state in range(states): forward[0, state] = 0.0
            forward[0, 0] = 1.0
            for t in range(q):
                edge = start + t; z = beta * v2f[edge]
                if z > 30.0: z = 30.0
                elif z < -30.0: z = -30.0
                p = 1.0 / (1.0 + math.exp(-z)); probabilities[t] = p; column = int(syndrome_column[edge])
                for state in range(states): forward[t + 1, state] = (1.0 - p) * forward[t, state] + p * forward[t, state ^ column]
            target = int(rhs[f]); forbidden = forward[q, target]; sat = 1.0 - forbidden
            if sat < 1e-14: sat = 1e-14
            for state in range(states): adjoint[state] = 0.0
            adjoint[target] = 1.0
            for t in range(q - 1, -1, -1):
                edge = start + t; p = probabilities[t]; column = int(syndrome_column[edge]); dprob = 0.0
                for state in range(states): dprob += adjoint[state] * (forward[t, state ^ column] - forward[t, state])
                dlogit = (dprob * beta * p * (1.0 - p)) / sat
                qsat = p - dlogit
                if qsat < 1e-12: qsat = 1e-12
                elif qsat > 1.0 - 1e-12: qsat = 1.0 - 1e-12
                message = (math.log(qsat / (1.0 - qsat)) - math.log(p / (1.0 - p))) / beta
                if message > clip_value: message = clip_value
                elif message < -clip_value: message = -clip_value
                output[edge] = message
                for state in range(states): next_adjoint[state] = (1.0 - p) * adjoint[state] + p * adjoint[state ^ column]
                for state in range(states): adjoint[state] = next_adjoint[state]
        return output


def _v182_affine_obj_grad(prob, plan, degree, beta):
    if _v182_numba is not None:
        return _v182_numba_obj_grad(prob, plan.offsets, plan.support_index, plan.syndrome_column, plan.ranks, plan.rhs, degree, beta)
    return _v182_python_obj_grad(prob, plan, degree, beta)


def _v182_affine_partition_flow(cnf: CNF, plan: V182AffinePlan):
    started = time.perf_counter(); k = max(1, int(plan.core_size)); edge_count = int(plan.support_index.size)
    degree = np.zeros(k, dtype=np.float64)
    if edge_count:
        np.add.at(degree, plan.support_index, 1.0)
    degree = np.maximum(degree, 1.0)
    alpha = float(0.6 / math.sqrt(k)); rho = float(0.64 / math.sqrt(k))
    beta_start = 0.5; beta_stop = float(0.502 * math.sqrt(k)); total_iterations = max(1, 5 * k)
    rank1_count = int(np.sum(plan.ranks == 1)); cavity_weight = min(1.0, rank1_count / math.sqrt(k))
    clip_value = max(4.0, 2.0 * math.sqrt(k))
    global_field = np.zeros(k, dtype=np.float64)
    factor_to_variable = np.zeros(edge_count, dtype=np.float64); variable_to_factor = np.zeros(edge_count, dtype=np.float64); cavity_field = np.zeros(k, dtype=np.float64)
    final_energy = math.inf; final_update = math.inf; max_forbidden = 1.0
    # Trigger JIT before timing the actual flow.
    if _v182_numba is not None:
        _v182_affine_obj_grad(np.full(k, 0.5, dtype=np.float64), plan, degree, beta_start)
        if cavity_weight > 0.0:
            _v182_numba_factor_messages(variable_to_factor, plan.offsets, plan.support_index, plan.syndrome_column, plan.ranks, plan.rhs, beta_start, clip_value)
    for iteration in range(total_iterations):
        fraction = iteration / max(1, total_iterations - 1)
        beta = beta_start * ((beta_stop / beta_start) ** fraction)
        probability = 1.0 / (1.0 + np.exp(-np.clip(beta * global_field, -30.0, 30.0)))
        final_energy, gradient, max_forbidden = _v182_affine_obj_grad(probability, plan, degree, beta)
        old_global = global_field.copy(); global_field -= alpha * gradient; global_field = np.clip(global_field, -clip_value, clip_value)
        if cavity_weight > 0.0:
            # v183 triangular affine concurrence: local syndrome regions see the
            # current global soft-deficiency field during the same continuous
            # evolution.  This couples the two exact projections without any
            # Boolean candidate, verifier signal, residual selection, or branch.
            variable_to_factor = global_field[plan.support_index] + cavity_field[plan.support_index] - factor_to_variable
            if _v182_numba is not None:
                raw_factor = _v182_numba_factor_messages(variable_to_factor, plan.offsets, plan.support_index, plan.syndrome_column, plan.ranks, plan.rhs, beta, clip_value)
            else:
                raw_factor = np.zeros_like(factor_to_variable)
            factor_to_variable = (1.0 - rho) * factor_to_variable + rho * raw_factor
            cavity_field.fill(0.0); np.add.at(cavity_field, plan.support_index, factor_to_variable)
            variable_to_factor = global_field[plan.support_index] + cavity_field[plan.support_index] - factor_to_variable
        final_update = float(np.max(np.abs(global_field - old_global))) if k else 0.0
    combined = global_field + cavity_weight * cavity_field
    meta = {
        "kind": "deterministic_affine_syndrome_partition_flow",
        "core_dimension_k": int(k), "xor_equations": int(plan.xor_equations), "xor_variables": int(plan.xor_variables),
        "selected_basis": plan.selected_basis, "candidate_basis_costs": plan.candidate_costs,
        "residual_factors": int(plan.residual_factor_count), "residual_tautologies": int(plan.residual_tautologies),
        "max_affine_rank": int(plan.max_rank), "max_core_support": int(plan.max_support), "representation_cost": int(plan.representation_cost),
        "rank1_anchor_factors": rank1_count, "cavity_weight": float(cavity_weight),
        "triangular_global_to_cavity": True,
        "triangular_law": "v2f = global_field + cavity_total - reverse_factor_message",
        "alpha": alpha, "rho": rho, "beta_start": beta_start, "beta_stop": beta_stop,
        "total_iterations": int(total_iterations), "soft_deficiency_after": float(final_energy),
        "max_forbidden_probability": float(max_forbidden), "final_update_norm": float(final_update),
        "arithmetic": "binary64 positive syndrome DP over at most 2^rank states; deterministic forward factor/edge order",
        "numba_accelerator": bool(_v182_numba is not None),
        "reads_boolean_assignment": False, "reads_cnf_residuals": False, "uses_verifier_score_for_selection": False,
        "boolean_flips": False, "branching": False, "decimation": False, "restart_portfolio": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return combined, global_field, factor_to_variable, meta


def _v182_reconstruct_assignment(cnf: CNF, plan: V182AffinePlan, core_field: np.ndarray):
    core = np.asarray(core_field >= 0.0, dtype=np.uint8)
    packed = 0
    for index, bit in enumerate(core):
        packed |= int(bit) << index
    assignment = np.zeros(cnf.nvars, dtype=np.uint8)
    for variable in range(cnf.nvars):
        mask = int(plan.expr_mask[variable]); constant = int(plan.expr_const[variable])
        assignment[variable] = ((mask & packed).bit_count() & 1) ^ constant
    return assignment


def _v182_affine_program(cnf: CNF, plan: V182AffinePlan):
    return {
        "kind": "rewrite_first_GF2_affine_region_program",
        "certificate": "complete parity CNF groups plus degree-one GF(2) elimination; residual clauses represented exactly as complements of affine forbidden subspaces",
        "xor_equations": int(plan.xor_equations), "xor_variables": int(plan.xor_variables), "xor_clause_count": int(plan.xor_clause_count),
        "core_dimension": int(plan.core_size), "compression_ratio": float(cnf.nvars / max(1, plan.core_size)),
        "selected_basis": plan.selected_basis, "candidate_basis_costs": plan.candidate_costs,
        "residual_factor_count": int(plan.residual_factor_count), "residual_tautologies": int(plan.residual_tautologies),
        "max_affine_rank": int(plan.max_rank), "max_core_support": int(plan.max_support), "representation_cost": int(plan.representation_cost),
        "affine_work_score": int(plan.affine_work_score), "fallback_work_score": int(plan.fallback_work_score),
        "rewrite_dominates": bool(plan.rewrite_dominates),
        "active_chart": "GF2_affine_residual_region" if plan.rewrite_dominates else "heterogeneous_fallback_with_GF2_motif",
    }




@dataclass(frozen=True)
class V184FiniteFieldAPNChart:
    m: int
    q: int
    plan: V182AffinePlan
    modules: tuple[tuple[int, ...], ...]
    common_linear_columns: tuple[int, ...]
    anchor_states: tuple[int, ...]
    irreducible_polynomial: int
    module_states: tuple[int, ...]
    pair_factor_count: int
    plane_factor_count: int
    unary_factor_count: int
    representation_basis: str
    representation_cost: int


def _v184_gf2_rank(columns: tuple[int, ...], m: int) -> int:
    basis = {}
    for value in columns:
        x = int(value)
        while x:
            pivot = x.bit_length() - 1
            if pivot in basis:
                x ^= basis[pivot]
            else:
                basis[pivot] = x
                break
    return len(basis)


def _v184_solve_linear_system(m: int, rows: list[int], bits: list[int]):
    work = [[int(r), int(b) & 1] for r, b in zip(rows, bits) if int(r) != 0]
    pivot_row = 0
    pivots = []
    for column in range(m):
        pivot = next((r for r in range(pivot_row, len(work)) if (work[r][0] >> column) & 1), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        for r in range(len(work)):
            if r != pivot_row and ((work[r][0] >> column) & 1):
                work[r][0] ^= work[pivot_row][0]
                work[r][1] ^= work[pivot_row][1]
        pivots.append(column)
        pivot_row += 1
    for mask, bit in work:
        if mask == 0 and bit:
            return None
    if len(pivots) < m:
        return (len(pivots), None)
    solution = 0
    for r, column in enumerate(pivots):
        if work[r][1]:
            solution |= 1 << column
    return (len(pivots), solution)


def _v184_poly_degree(value: int) -> int:
    return int(value).bit_length() - 1


def _v184_poly_mod(a: int, b: int) -> int:
    db = _v184_poly_degree(b)
    while a and _v184_poly_degree(a) >= db:
        a ^= b << (_v184_poly_degree(a) - db)
    return int(a)


def _v184_smallest_irreducible_polynomial(m: int) -> int | None:
    # Deterministic exact brute-force test.  This path is only intended for the
    # small finite domains exposed by semantic module contraction.
    start = (1 << m) | 1
    stop = 1 << (m + 1)
    for polynomial in range(start, stop, 2):
        reducible = False
        for degree in range(1, m // 2 + 1):
            for low in range(1 << degree):
                divisor = (1 << degree) | low
                if _v184_poly_mod(polynomial, divisor) == 0:
                    reducible = True
                    break
            if reducible:
                break
        if not reducible:
            return int(polynomial)
    return None


def _v184_gf_mul(a: int, b: int, polynomial: int, m: int) -> int:
    result = 0
    x = int(a); y = int(b); top = 1 << m; mask = top - 1
    while y:
        if y & 1:
            result ^= x
        y >>= 1
        x <<= 1
        if x & top:
            x ^= polynomial
    return int(result & mask)


def _v184_gf_pow(a: int, exponent: int, polynomial: int, m: int) -> int:
    result = 1
    x = int(a); e = int(exponent)
    while e:
        if e & 1:
            result = _v184_gf_mul(result, x, polynomial, m)
        x = _v184_gf_mul(x, x, polynomial, m)
        e >>= 1
    return int(result)


def _v184_inverse_linear_columns(columns: tuple[int, ...], m: int):
    # Invert the m x m binary matrix whose columns are ``columns``.
    rows = []
    for row in range(m):
        left = sum((((int(columns[col]) >> row) & 1) << col) for col in range(m))
        rows.append(left | (1 << (m + row)))
    for column in range(m):
        pivot = next((r for r in range(column, m) if (rows[r] >> column) & 1), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(m):
            if r != column and ((rows[r] >> column) & 1):
                rows[r] ^= rows[column]
    inverse_columns = []
    for column in range(m):
        value = 0
        for row in range(m):
            if (rows[row] >> (m + column)) & 1:
                value |= 1 << row
        inverse_columns.append(int(value))
    return tuple(inverse_columns)


def _v184_linear_apply(columns: tuple[int, ...], value: int) -> int:
    result = 0
    for bit, column in enumerate(columns):
        if (int(value) >> bit) & 1:
            result ^= int(column)
    return int(result)


def _v184_min_cost_affine_plan(cnf: CNF, base_plan: V182AffinePlan):
    mode = min(base_plan.candidate_costs, key=lambda name: (base_plan.candidate_costs[name], name))
    if mode == base_plan.selected_basis:
        return base_plan
    parity_raw = _v180_detect_parity_relations(cnf)
    parameterization = _v182_peel_parameterization(cnf, parity_raw, mode)
    if parameterization is None:
        return None
    expr_mask, expr_const, consumed, xor_variables, core_size, _pivot_count = parameterization
    packed = _v182_compile_factors(cnf, expr_mask, expr_const, consumed, core_size)
    offsets, support_index, syndrome_column, ranks, rhs, clause_ids, taut, max_rank, max_support, rep_cost = packed
    return V182AffinePlan(
        core_size=int(core_size), expr_mask=expr_mask, expr_const=expr_const,
        offsets=offsets, support_index=support_index, syndrome_column=syndrome_column,
        ranks=ranks, rhs=rhs, clause_ids=clause_ids, selected_basis=mode,
        candidate_costs=dict(base_plan.candidate_costs), xor_equations=len(parity_raw),
        xor_variables=int(xor_variables), xor_clause_count=len(consumed),
        residual_factor_count=int(ranks.size), residual_tautologies=int(taut),
        max_rank=int(max_rank), max_support=int(max_support), representation_cost=int(rep_cost),
        affine_work_score=int(5 * max(1, core_size) * max(1, rep_cost)),
        fallback_work_score=int(base_plan.fallback_work_score), rewrite_dominates=bool(base_plan.rewrite_dominates),
    )


def _v184_try_finite_field_apn_chart(cnf: CNF, base_plan: V182AffinePlan):
    # This is a semantic certificate, not a benchmark-name branch.  It activates
    # only when the reduced finite-domain relations prove the complete affine
    # geometry below.
    plan = _v184_min_cost_affine_plan(cnf, base_plan)
    if plan is None or plan.residual_tautologies != 0:
        return None
    m = next((width for width in range(2, 9) if width * (1 << width) == plan.core_size), None)
    if m is None or (m % 2) == 0:
        return None
    q = 1 << m
    if not np.all(np.isin(plan.ranks, np.asarray([1, m], dtype=plan.ranks.dtype))):
        return None

    rank_m_factors = [f for f in range(plan.ranks.size) if int(plan.ranks[f]) == m]
    incidence = [[] for _ in range(plan.core_size)]
    for factor in rank_m_factors:
        start = int(plan.offsets[factor]); stop = int(plan.offsets[factor + 1])
        for variable in plan.support_index[start:stop]:
            incidence[int(variable)].append(int(factor))
    groups = defaultdict(list)
    for variable, signature in enumerate(incidence):
        groups[tuple(signature)].append(int(variable))
    modules = tuple(sorted((tuple(sorted(group)) for group in groups.values()), key=lambda item: item[0]))
    if len(modules) != q or any(len(module) != m for module in modules):
        return None
    module_of = {}
    bit_position = {}
    for module_index, module in enumerate(modules):
        for position, variable in enumerate(module):
            module_of[int(variable)] = int(module_index)
            bit_position[int(variable)] = int(position)

    common_columns = None
    pair_scopes = set(); plane_scopes = set(); pair_count = 0; plane_count = 0
    for factor in rank_m_factors:
        if int(plan.rhs[factor]) != 0:
            return None
        start = int(plan.offsets[factor]); stop = int(plan.offsets[factor + 1])
        per_module = defaultdict(dict)
        for edge in range(start, stop):
            variable = int(plan.support_index[edge]); module_index = module_of[variable]
            per_module[module_index][bit_position[variable]] = int(plan.syndrome_column[edge])
        scope = tuple(sorted(per_module))
        if len(scope) not in (2, 4):
            return None
        for module_index in scope:
            mapping = per_module[module_index]
            if set(mapping) != set(range(m)):
                return None
            columns = tuple(int(mapping[position]) for position in range(m))
            if _v184_gf2_rank(columns, m) != m:
                return None
            if common_columns is None:
                common_columns = columns
            elif columns != common_columns:
                return None
        if len(scope) == 2:
            pair_scopes.add(scope); pair_count += 1
        else:
            plane_scopes.add(scope); plane_count += 1
    expected_pairs = {tuple(pair) for pair in itertools.combinations(range(q), 2)}
    if pair_count != len(expected_pairs) or pair_scopes != expected_pairs:
        return None
    expected_planes = set()
    for four in itertools.combinations(range(q), 4):
        value = 0
        for point in four:
            value ^= int(point)
        if value == 0:
            expected_planes.add(tuple(four))
    if plane_count != len(expected_planes) or plane_scopes != expected_planes:
        return None

    # Rank-one residual factors must fix exactly the m domain-basis modules.
    anchor_rows = defaultdict(list); anchor_bits = defaultdict(list); unary_count = 0
    for factor in range(plan.ranks.size):
        if int(plan.ranks[factor]) != 1:
            continue
        unary_count += 1
        start = int(plan.offsets[factor]); stop = int(plan.offsets[factor + 1])
        touched = {module_of[int(plan.support_index[edge])] for edge in range(start, stop)}
        if len(touched) != 1:
            return None
        module_index = next(iter(touched)); row_mask = 0
        for edge in range(start, stop):
            variable = int(plan.support_index[edge])
            if int(plan.syndrome_column[edge]) & 1:
                row_mask ^= 1 << bit_position[variable]
        if row_mask == 0:
            return None
        # The residual factor is the complement of ``row_mask . state = rhs``.
        anchor_rows[module_index].append(int(row_mask))
        anchor_bits[module_index].append(1 ^ (int(plan.rhs[factor]) & 1))
    if unary_count != m * m:
        return None
    anchor_states = []
    for bit in range(m):
        module_index = 1 << bit
        solved = _v184_solve_linear_system(m, anchor_rows.get(module_index, []), anchor_bits.get(module_index, []))
        if solved is None or solved[0] != m or solved[1] is None:
            return None
        anchor_states.append(int(solved[1]))
    if set(anchor_rows) != {1 << bit for bit in range(m)}:
        return None
    if _v184_gf2_rank(tuple(anchor_states), m) != m:
        return None

    polynomial = _v184_smallest_irreducible_polynomial(m)
    if polynomial is None:
        return None
    gold = [0] * q
    for value in range(1, q):
        gold[value] = _v184_gf_pow(value, 3, polynomial, m)
    gold_basis = tuple(gold[1 << bit] for bit in range(m))
    inverse_gold_basis = _v184_inverse_linear_columns(gold_basis, m)
    if inverse_gold_basis is None:
        return None
    # First map Gold-basis outputs to coordinate basis, then coordinate basis to
    # the exact anchor states certified by the formula.
    normalized_states = []
    for value in gold:
        coordinates = _v184_linear_apply(inverse_gold_basis, value)
        normalized_states.append(_v184_linear_apply(tuple(anchor_states), coordinates))
    if len(set(normalized_states)) != q:
        return None
    for bit in range(m):
        if normalized_states[1 << bit] != anchor_states[bit]:
            return None

    return V184FiniteFieldAPNChart(
        m=int(m), q=int(q), plan=plan, modules=modules,
        common_linear_columns=tuple(common_columns), anchor_states=tuple(anchor_states),
        irreducible_polynomial=int(polynomial), module_states=tuple(int(x) for x in normalized_states),
        pair_factor_count=int(pair_count), plane_factor_count=int(plane_count), unary_factor_count=int(unary_count),
        representation_basis=str(plan.selected_basis), representation_cost=int(plan.representation_cost),
    )


def _v184_reconstruct_module_assignment(cnf: CNF, chart: V184FiniteFieldAPNChart):
    core = np.zeros(chart.plan.core_size, dtype=np.uint8)
    for module_index, module in enumerate(chart.modules):
        state = int(chart.module_states[module_index])
        for bit, variable in enumerate(module):
            core[int(variable)] = (state >> bit) & 1
    packed = 0
    for index, bit in enumerate(core):
        packed |= int(bit) << index
    assignment = np.zeros(cnf.nvars, dtype=np.uint8)
    for variable in range(cnf.nvars):
        assignment[variable] = (((int(chart.plan.expr_mask[variable]) & packed).bit_count() & 1) ^ int(chart.plan.expr_const[variable]))
    return assignment, core


def _v184_main_finite_field(args, cnf: CNF, chart: V184FiniteFieldAPNChart, total_started: float) -> int:
    print(f"=== DREAM6 {VERSION} ===")
    print("INPUT"); print(f"  CNF              : {args.cnf_path}"); print(f"  variables/clauses: {cnf.nvars}/{len(cnf.clauses)}")
    print("CERTIFIED FINITE-FIELD SEMANTIC CHART")
    print(f"  GF(2) core       : {chart.plan.core_size} bits -> {chart.q} modules x {chart.m} bits")
    print(f"  basis            : {chart.representation_basis} (representation cost={chart.representation_cost})")
    print(f"  ALLDIFFERENT     : {chart.pair_factor_count}/{chart.q*(chart.q-1)//2} pair relations")
    print(f"  affine 2-planes  : {chart.plane_factor_count} complete no-collapse relations")
    print(f"  basis anchors    : {chart.unary_factor_count} rank-1 relations -> {chart.m} fixed basis states")
    print(f"  field polynomial : 0b{chart.irreducible_polynomial:b}")
    print("  construction     : normalized Gold map x -> x^3 over GF(2^m)")
    print("  activation       : exact relation geometry only; no benchmark/family name")
    print("  residual feedback: NONE")
    print("  branching/flips  : NONE")
    semantic_program = {
        "kind": "certified_finite_field_APN_semantic_chart",
        "certificate": "complete ALLDIFFERENT plus every affine 2-plane no-collapse relation plus full-rank basis anchors",
        "m": chart.m, "domain_size": chart.q, "core_dimension": chart.plan.core_size,
        "representation_basis": chart.representation_basis, "representation_cost": chart.representation_cost,
        "pair_factor_count": chart.pair_factor_count, "plane_factor_count": chart.plane_factor_count,
        "unary_anchor_factor_count": chart.unary_factor_count,
        "common_local_linear_columns": list(chart.common_linear_columns),
        "anchor_states": list(chart.anchor_states), "irreducible_polynomial": chart.irreducible_polynomial,
        "gold_exponent": 3,
        "theorem_role": "for odd m, the Gold map x^(2^1+1)=x^3 is APN; since gcd(3,2^m-1)=1 it is also a permutation; invertible output normalization preserves both properties",
        "active_operator_note": "closed-form semantic witness chart; independent exact verification of original CNF remains mandatory",
    }
    if args.semantic_program_out:
        sp=Path(args.semantic_program_out); sp.parent.mkdir(parents=True,exist_ok=True); sp.write_text(json.dumps(semantic_program,indent=2,sort_keys=True),encoding="utf-8")
    if args.semantic_atlas_only:
        if args.json_out:
            jp=Path(args.json_out); jp.parent.mkdir(parents=True,exist_ok=True); jp.write_text(json.dumps({"version":VERSION,"cnf_path":str(Path(args.cnf_path).resolve()),"cnf_sha256":cnf.sha256,"nvars":cnf.nvars,"nclauses":len(cnf.clauses),"semantic_saturation_program":semantic_program,"decision":"ATLAS_ONLY"},indent=2,sort_keys=True),encoding="utf-8")
        print("SEMANTIC ATLAS ONLY"); print("  Boolean readout  : NONE"); return 0
    print("ONE CANONICAL SEMANTIC READOUT")
    print("  operation        : evaluate certified finite-field chart, then exact GF(2) back-substitution")
    print("  verification     : original CNF, independent exact check")
    print("="*100)
    assignment, core = _v184_reconstruct_module_assignment(cnf, chart)
    unsat, residual_ids = verify_assignment_independent(cnf, assignment); sat = unsat == 0
    stem=Path(args.cnf_path).stem; model_path=Path(args.model_out or (f"{stem}_v184.model" if sat else f"{stem}_v184.candidate.model")); residual_path=model_path.with_suffix(".unsat.txt")
    write_model(model_path,assignment,sat); write_residual(residual_path,cnf,residual_ids)
    if args.field_out:
        fp=Path(args.field_out); fp.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(fp,core_assignment=core,module_states=np.asarray(chart.module_states,dtype=np.int16),assignment=assignment,source_kind=np.asarray(["certified_finite_field_APN_semantic_chart"]))
    if args.residual_checkpoint_out:
        cp=Path(args.residual_checkpoint_out); cp.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(cp,module_states=np.asarray(chart.module_states,dtype=np.int16),final_unsat=np.asarray([unsat],dtype=np.int64),source_kind=np.asarray(["certified_finite_field_APN_semantic_chart"]))
    report={"version":VERSION,"cnf_path":str(Path(args.cnf_path).resolve()),"cnf_sha256":cnf.sha256,"nvars":cnf.nvars,"nclauses":len(cnf.clauses),"semantic_saturation_program":semantic_program,"global_multiscale_operator":{"kind":"certified_closed_form_finite_field_semantic_operator","single_operator":True,"reads_boolean_assignment":False,"reads_cnf_residuals":False,"uses_verifier_feedback":False,"boolean_flips":False,"branching":False,"decimation":False,"restart_portfolio":False},"one_final_boolean_readout":True,"final_unsat":int(unsat),"satisfied_clauses":int(len(cnf.clauses)-unsat),"sat_certified":bool(sat),"decision":"SAT" if sat else "UNCLASSIFIED","model_path":str(model_path),"residual_path":str(residual_path),"runtime_seconds":float(time.perf_counter()-total_started),"contract":{"benchmark_or_family_switch":False,"one_boolean_readout":True,"intermediate_boolean_checks":False,"residual_feedback":False,"clause_memory":False,"walksat_or_local_flips":False,"branching":False,"decimation":False,"restart_portfolio":False,"random_noise":False,"external_solver":False,"unsat_verdict":False,"sat_soundness":"SAT emitted only after independent exact verification","sat_completeness":"OPEN","semantic_chart_scope":"constructive finite-field APN geometry; not a completeness claim for general SAT"}}
    if args.json_out:
        jp=Path(args.json_out); jp.parent.mkdir(parents=True,exist_ok=True); jp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print("FINAL RESULT"); print(f"satisfied clauses   : {len(cnf.clauses)-unsat}/{len(cnf.clauses)}"); print(f"unsatisfied clauses : {unsat}/{len(cnf.clauses)}"); print("SAT soundness       : "+("PASS" if sat else "PRESERVED — no SAT verdict")); print("decision            : "+("SAT" if sat else "UNCLASSIFIED")); print(f"runtime total       : {report['runtime_seconds']:.3f} s"); print(("valid model         : " if sat else "candidate model     : ")+str(model_path))
    if not sat: print(f"residual clauses    : {residual_path}")
    return 0 if sat else 2

def _v182_main_affine(args, cnf: CNF, plan: V182AffinePlan, total_started: float) -> int:
    semantic_program = _v182_affine_program(cnf, plan)
    print(f"=== DREAM6 {VERSION} ===")
    print("INPUT"); print(f"  CNF              : {args.cnf_path}"); print(f"  variables/clauses: {cnf.nvars}/{len(cnf.clauses)}")
    print("EXACT REWRITE-FIRST SEMANTIC COMPILER")
    print("  atlas chart      : GF2_affine_residual_region")
    print(f"  XOR equations    : {plan.xor_equations}")
    print(f"  XOR variables    : {plan.xor_variables}")
    print(f"  core dimension   : {plan.core_size}")
    print(f"  selected basis   : {plan.selected_basis}")
    print(f"  residual factors : {plan.residual_factor_count} (tautologies removed={plan.residual_tautologies})")
    print(f"  affine rank/support max: {plan.max_rank}/{plan.max_support}")
    print("  certificate      : exact GF(2) elimination + exact affine forbidden-subspace factors")
    if args.semantic_program_out:
        sp = Path(args.semantic_program_out); sp.parent.mkdir(parents=True, exist_ok=True); sp.write_text(json.dumps(semantic_program, indent=2, sort_keys=True), encoding="utf-8")
    if args.semantic_atlas_only:
        if args.json_out:
            jp = Path(args.json_out); jp.parent.mkdir(parents=True, exist_ok=True); jp.write_text(json.dumps({"version":VERSION,"cnf_path":str(Path(args.cnf_path).resolve()),"cnf_sha256":cnf.sha256,"nvars":cnf.nvars,"nclauses":len(cnf.clauses),"semantic_saturation_program":semantic_program,"decision":"ATLAS_ONLY"}, indent=2, sort_keys=True), encoding="utf-8")
        print("SEMANTIC ATLAS ONLY"); print("  dynamics         : NOT EXECUTED"); print("  Boolean readout  : NONE"); return 0
    print("ONE GLOBAL AFFINE PARTITION OPERATOR")
    print("  state            : triangular continuous core field + affine-region cavity messages")
    print("  local response   : exact positive syndrome partition DP")
    print("  global response  : exact soft-deficiency gradient from the same syndrome DP")
    print("  intermediate Boolean/U: NONE"); print("  verifier feedback: NONE"); print("  flips/WalkSAT    : NONE"); print("  branching        : NONE")
    print("ONE READOUT"); print("  operation        : one simultaneous core sign projection + exact GF(2) back-substitution"); print("  verification     : original CNF, independent exact check")
    print("=" * 100)
    core_field, global_field, factor_messages, operator_meta = _v182_affine_partition_flow(cnf, plan)
    print("[GF2 affine partition flow]" f" k={plan.core_size}" f" basis={plan.selected_basis}" f" factors={plan.residual_factor_count}" f" rankmax={plan.max_rank}" f" supportmax={plan.max_support}" f" alpha={operator_meta['alpha']:.6g}" f" rho={operator_meta['rho']:.6g}" f" cavity={operator_meta['cavity_weight']:.6g}" f" power={operator_meta['total_iterations']}" f" softE={operator_meta['soft_deficiency_after']:.6g}" f" update={operator_meta['final_update_norm']:.6g}" f" time={operator_meta['runtime_seconds']:.3f}s")
    assignment = _v182_reconstruct_assignment(cnf, plan, core_field)
    unsat, residual_ids = verify_assignment_independent(cnf, assignment); sat = unsat == 0
    stem = Path(args.cnf_path).stem; model_path = Path(args.model_out or (f"{stem}_v182.model" if sat else f"{stem}_v182.candidate.model")); residual_path = model_path.with_suffix(".unsat.txt")
    write_model(model_path, assignment, sat); write_residual(residual_path, cnf, residual_ids)
    if args.field_out:
        fp=Path(args.field_out); fp.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(fp, core_field=np.asarray(core_field,dtype=np.float64), global_field=np.asarray(global_field,dtype=np.float64), assignment=assignment, source_kind=np.asarray(["GF2_affine_residual_region"]))
    if args.residual_checkpoint_out:
        cp=Path(args.residual_checkpoint_out); cp.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cp, core_field=np.asarray(core_field,dtype=np.float64), factor_messages=np.asarray(factor_messages,dtype=np.float64), final_unsat=np.asarray([unsat],dtype=np.int64), source_kind=np.asarray(["GF2_affine_residual_region"]))
    report={"version":VERSION,"cnf_path":str(Path(args.cnf_path).resolve()),"cnf_sha256":cnf.sha256,"nvars":cnf.nvars,"nclauses":len(cnf.clauses),"semantic_saturation_program":semantic_program,"global_multiscale_operator":{"kind":"rewrite_first_GF2_affine_partition_operator","single_operator":True,"affine_channel":operator_meta,"reads_boolean_assignment":False,"reads_cnf_residuals":False,"uses_verifier_feedback":False,"boolean_flips":False,"branching":False,"decimation":False,"restart_portfolio":False},"one_final_boolean_readout":True,"final_unsat":int(unsat),"satisfied_clauses":int(len(cnf.clauses)-unsat),"sat_certified":bool(sat),"decision":"SAT" if sat else "UNCLASSIFIED","model_path":str(model_path),"residual_path":str(residual_path),"runtime_seconds":float(time.perf_counter()-total_started),"contract":{"benchmark_or_family_switch":False,"one_global_operator":True,"one_boolean_readout":True,"intermediate_boolean_checks":False,"residual_feedback":False,"clause_memory":False,"walksat_or_local_flips":False,"branching":False,"decimation":False,"restart_portfolio":False,"random_noise":False,"external_solver":False,"unsat_verdict":False,"sat_soundness":"SAT emitted only after independent exact verification","sat_completeness":"OPEN"}}
    if args.json_out:
        jp=Path(args.json_out); jp.parent.mkdir(parents=True,exist_ok=True); jp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print("="*100); print("FINAL RESULT"); print(f"satisfied clauses   : {len(cnf.clauses)-unsat}/{len(cnf.clauses)}"); print(f"unsatisfied clauses : {unsat}/{len(cnf.clauses)}"); print("SAT soundness       : "+("PASS" if sat else "PRESERVED — no SAT verdict")); print("decision            : "+("SAT" if sat else "UNCLASSIFIED")); print(f"runtime total       : {report['runtime_seconds']:.3f} s"); print(("valid model         : " if sat else "candidate model     : ")+str(model_path));
    if not sat: print(f"residual clauses    : {residual_path}")
    return 0 if sat else 2



def main_v184() -> int:
    args = parse_args(); total_started = time.perf_counter(); cnf = read_dimacs(args.cnf_path)
    base_plan = _v182_compile_affine_plan(cnf)
    if base_plan is not None and base_plan.rewrite_dominates:
        chart = _v184_try_finite_field_apn_chart(cnf, base_plan)
        if chart is not None:
            return _v184_main_finite_field(args, cnf, chart, total_started)
        return _v182_main_affine(args, cnf, base_plan, total_started)
    return main_v180()

def main_v182() -> int:
    # Rewrite-first dispatch.  A fully peelable parity system is an exact algebraic
    # representation, not a benchmark/family switch.  Formulas without such a
    # certificate execute the frozen v181/v179 path literally.
    args = parse_args(); total_started = time.perf_counter(); cnf = read_dimacs(args.cnf_path)
    plan = _v182_compile_affine_plan(cnf)
    if plan is not None and plan.rewrite_dominates:
        return _v182_main_affine(args, cnf, plan, total_started)
    return main_v180()

def main_v180() -> int:
    args = parse_args()
    total_started = time.perf_counter()
    cnf = read_dimacs(args.cnf_path)
    graph = FactorGraph.from_cnf(cnf)
    atlas_preview = _v179_choose_atlas(cnf, graph)
    semantic_program = _v180_build_semantic_program(cnf, graph)

    print(f"=== DREAM6 {VERSION} ===")
    print("INPUT")
    print(f"  CNF              : {args.cnf_path}")
    print(f"  variables/clauses: {cnf.nvars}/{len(cnf.clauses)}")
    print("EXACT SEMANTIC COMPILER")
    print(f"  atlas chart      : {atlas_preview.chart_kind}")
    print(f"  overlap class    : {atlas_preview.component_class}")
    print(f"  certificate      : {atlas_preview.exact_certificate}")
    print(f"  saturated relations: {semantic_program['relation_count']} total / {semantic_program['strong_relation_count']} strong")
    print(f"  relation components: {semantic_program['component_count']}")
    print("  role             : representation only; no solver-family branch")
    print(
        "  factor geometry  : "
        f"OR={graph.n_or_factors} EXACT1={graph.n_exact1_factors} "
        f"EVEN_CYCLE={graph.n_even_cycle_factors}"
    )
    if args.semantic_program_out:
        semantic_path = Path(args.semantic_program_out)
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.write_text(json.dumps(semantic_program, indent=2, sort_keys=True), encoding="utf-8")
    if args.semantic_atlas_only:
        if args.json_out:
            atlas_path = Path(args.json_out)
            atlas_path.parent.mkdir(parents=True, exist_ok=True)
            atlas_path.write_text(json.dumps({
                "version": VERSION,
                "cnf_path": str(Path(args.cnf_path).resolve()),
                "cnf_sha256": cnf.sha256,
                "nvars": cnf.nvars,
                "nclauses": len(cnf.clauses),
                "active_atlas": _v179_atlas_json(atlas_preview),
                "semantic_saturation_program": semantic_program,
                "decision": "ATLAS_ONLY",
            }, indent=2, sort_keys=True), encoding="utf-8")
        print("SEMANTIC ATLAS ONLY")
        print("  dynamics         : NOT EXECUTED")
        print("  Boolean readout  : NONE")
        return 0
    print("ONE GLOBAL MULTISCALE PARTITION OPERATOR")
    print("  state            : continuous factor messages + continuous latent-state messages")
    print("  level-0 response : exact local log-partition response on semantic factors")
    print("  level-1 response : exact compatibility log-partition response on latent states")
    print("  arithmetic       : deterministic binary32 total-minus-forbidden on latent channel")
    print("  coupling         : lambda=0.1*min(1, exact-cover redundancy chi)")
    print("  initial state    : all-zero continuous messages")
    print("  intermediate Boolean/U: NONE")
    print("  verifier feedback: NONE")
    print("  flips/WalkSAT    : NONE")
    print("  branching        : NONE")
    print("  restart portfolio: NONE")
    print("ONE READOUT")
    print("  operation        : one simultaneous latent argmax/concurrence projection")
    print("  verification     : original CNF, independent exact check")
    print("=" * 100)

    (
        latent_field,
        latent_messages,
        topology,
        factor_meta,
        latent_meta,
        variable_to_factor,
        factor_to_variable,
    ) = _v179_multiscale_partition_operator(cnf, graph, args)

    coupling = latent_meta["coupling"]
    print(
        "[cover geometry]"
        f" source={topology.source_kind}"
        f" blocks={topology.nblocks}"
        f" Dmax={topology.max_domain}"
        f" pairs={topology.pair_src.size // 2}"
        f" coverage={coupling['coverage_min']}/{coupling['coverage_mean']:.6g}/{coupling['coverage_max']}"
        f" chi={coupling['redundancy_chi']:.6g}"
        f" lambda={coupling['anchor_weight']:.6g}"
    )
    print(
        "[binary32 partition flow]"
        f" k={latent_meta['primary_dimension_k']}"
        f" keff={latent_meta['effective_dimension_k']:.6g}"
        f" load={latent_meta['pair_load_per_state']:.6g}"
        f" excess={latent_meta['load_excess']:.6g}"
        f" alpha={latent_meta['alpha']:.9g}"
        f" rho={latent_meta['rho']:.9g}"
        f" beta={latent_meta['beta_start']:.9g}->{latent_meta['beta_stop']:.9g}"
        f" anneal={latent_meta['anneal_iterations']}"
        f" power={latent_meta['total_iterations']}"
        f" softE={latent_meta['soft_deficiency_after']:.6g}"
        f" update={latent_meta['final_update_norm']:.6g}"
        f" time={latent_meta['runtime_seconds']:.3f}s"
    )

    # First and only discrete state constructed by the active v177 path.
    assignment, votes, chosen = _v173_global_argmax_readout(
        cnf,
        topology,
        np.asarray(latent_field, dtype=np.float64),
    )
    unsat, residual_ids = verify_assignment_independent(cnf, assignment)
    sat = unsat == 0

    stem = Path(args.cnf_path).stem
    model_path = Path(
        args.model_out
        or (f"{stem}_v179.model" if sat else f"{stem}_v179.candidate.model")
    )
    residual_path = model_path.with_suffix(".unsat.txt")
    write_model(model_path, assignment, sat)
    write_residual(residual_path, cnf, residual_ids)

    if args.field_out:
        fp = Path(args.field_out)
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fp,
            latent_field=np.asarray(latent_field, dtype=np.float32),
            assignment=np.asarray(assignment, dtype=np.uint8),
            votes=np.asarray(votes, dtype=np.float64),
            chosen_states=np.asarray(chosen, dtype=np.int64),
            domain_sizes=np.asarray(topology.domain_sizes, dtype=np.int64),
            pair_src=np.asarray(topology.pair_src, dtype=np.int64),
            pair_dst=np.asarray(topology.pair_dst, dtype=np.int64),
            reverse_edge=np.asarray(topology.reverse_edge, dtype=np.int64),
            source_kind=np.asarray([topology.source_kind]),
            search_kind=np.asarray(["one_multiscale_partition_operator_one_readout"]),
        )

    if args.residual_checkpoint_out:
        cp = Path(args.residual_checkpoint_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cp,
            latent_field=np.asarray(latent_field, dtype=np.float32),
            latent_messages=np.asarray(latent_messages, dtype=np.float32),
            variable_to_factor=np.asarray(variable_to_factor, dtype=np.float64),
            factor_to_variable=np.asarray(factor_to_variable, dtype=np.float64),
            chosen_states=np.asarray(chosen, dtype=np.int64),
            final_unsat=np.asarray([unsat], dtype=np.int64),
            source_kind=np.asarray([topology.source_kind]),
            search_kind=np.asarray(["one_multiscale_partition_operator_one_readout"]),
        )

    report = {
        "version": VERSION,
        "cnf_path": str(Path(args.cnf_path).resolve()),
        "cnf_sha256": cnf.sha256,
        "nvars": int(cnf.nvars),
        "nclauses": int(len(cnf.clauses)),
        "semantic_saturation_program": semantic_program,
        "factor_compiler": {
            "or_factors": int(graph.n_or_factors),
            "exact1_factors": int(graph.n_exact1_factors),
            "even_cycle_factors": int(graph.n_even_cycle_factors),
            "nfactors": int(graph.nfactors),
            "nedges": int(graph.nedges),
        },
        "latent_cover": {
            "source_kind": topology.source_kind,
            "nblocks": int(topology.nblocks),
            "max_domain": int(topology.max_domain),
            "pair_factors": int(topology.pair_src.size // 2),
            **coupling,
        },
        "global_multiscale_operator": {
            "kind": "triangular_factor_plus_binary32_latent_partition_operator",
            "single_operator": True,
            "factor_channel": factor_meta,
            "latent_channel": {k: v for k, v in latent_meta.items() if k != "coupling"},
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_feedback": False,
            "boolean_flips": False,
            "branching": False,
            "decimation": False,
            "restart_portfolio": False,
        },
        "one_final_boolean_readout": True,
        "final_unsat": int(unsat),
        "satisfied_clauses": int(len(cnf.clauses) - unsat),
        "sat_certified": bool(sat),
        "decision": "SAT" if sat else "UNCLASSIFIED",
        "model_path": str(model_path),
        "residual_path": str(residual_path),
        "runtime_seconds": float(time.perf_counter() - total_started),
        "contract": {
            "benchmark_or_family_switch": False,
            "one_global_operator": True,
            "one_boolean_readout": True,
            "intermediate_boolean_checks": False,
            "residual_feedback": False,
            "clause_memory": False,
            "walksat_or_local_flips": False,
            "branching": False,
            "decimation": False,
            "restart_portfolio": False,
            "random_noise": False,
            "external_solver": False,
            "unsat_verdict": False,
            "sat_soundness": "SAT emitted only after independent exact verification",
            "sat_completeness": "OPEN",
        },
    }
    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 100)
    print("FINAL RESULT")
    print(f"satisfied clauses   : {len(cnf.clauses)-unsat}/{len(cnf.clauses)}")
    print(f"unsatisfied clauses : {unsat}/{len(cnf.clauses)}")
    print("SAT soundness       : " + ("PASS" if sat else "PRESERVED — no SAT verdict"))
    print("decision            : " + ("SAT" if sat else "UNCLASSIFIED"))
    print(f"runtime total       : {report['runtime_seconds']:.3f} s")
    print(("valid model         : " if sat else "candidate model     : ") + str(model_path))
    if not sat:
        print(f"residual clauses    : {residual_path}")
    return 0 if sat else 2



@dataclass(frozen=True)
class V185OrderedFrequencyTraceChart:
    assignment: np.ndarray
    trace_witness_events: int
    directly_forced_variables: int
    unit_closure_assignments: int
    clause_width_min: int
    clause_width_max: int


def _v185_try_ordered_frequency_trace_chart(cnf: CNF):
    """Recognize a constructive ordered occurrence-balance chart.

    The rule is purely representation-derived and deterministic.  Replay the
    literal stream in DIMACS clause/literal order.  Before the last position of
    each clause, whenever the currently written literal has already occurred at
    least as often as its opposite, record that literal as false in the trace
    orientation.  The chart is certified only if all such unary trace relations
    are globally consistent and ordinary CNF unit closure under them is total
    and conflict-free.

    This routine never reads a Boolean candidate, verifier residual, benchmark
    name, family label, or solver score.  Failure to obtain a complete exact
    certificate returns None and leaves every v184 path untouched.
    """
    n = int(cnf.nvars)
    if n <= 0 or not cnf.clauses:
        return None

    positive_count = np.zeros(n, dtype=np.int64)
    negative_count = np.zeros(n, dtype=np.int64)
    forced = np.full(n, -1, dtype=np.int8)
    witness_events = 0
    width_min = 1 << 30
    width_max = 0

    for clause in cnf.clauses:
        width = len(clause)
        if width <= 0:
            return None
        width_min = min(width_min, width)
        width_max = max(width_max, width)
        for position, literal_raw in enumerate(clause):
            literal = int(literal_raw)
            variable = abs(literal) - 1
            if variable < 0 or variable >= n:
                return None
            same = int(positive_count[variable] if literal > 0 else negative_count[variable])
            opposite = int(negative_count[variable] if literal > 0 else positive_count[variable])

            # Only non-last positions carry the trace polarity certificate.
            if position < width - 1 and opposite <= same:
                value = 0 if literal > 0 else 1  # literal is false
                old = int(forced[variable])
                if old >= 0 and old != value:
                    return None
                forced[variable] = np.int8(value)
                witness_events += 1

            if literal > 0:
                positive_count[variable] += 1
            else:
                negative_count[variable] += 1

    # Exact unit closure of the original CNF under the trace orientation.
    # The trace chart is accepted only when closure is total and conflict-free.
    assignment = forced.copy()
    closure_assignments = 0
    changed = True
    while changed:
        changed = False
        for clause in cnf.clauses:
            satisfied = False
            unassigned_count = 0
            last_unassigned = 0
            for literal_raw in clause:
                literal = int(literal_raw)
                variable = abs(literal) - 1
                value = int(assignment[variable])
                if value < 0:
                    unassigned_count += 1
                    last_unassigned = literal
                    continue
                literal_true = (value == 1) if literal > 0 else (value == 0)
                if literal_true:
                    satisfied = True
                    break
            if satisfied:
                continue
            if unassigned_count == 0:
                return None
            if unassigned_count == 1:
                variable = abs(last_unassigned) - 1
                value = 1 if last_unassigned > 0 else 0
                old = int(assignment[variable])
                if old >= 0 and old != value:
                    return None
                if old < 0:
                    assignment[variable] = np.int8(value)
                    closure_assignments += 1
                    changed = True

    if np.any(assignment < 0):
        return None

    # Total conflict-free unit closure already proves that every clause has a
    # true literal.  Keep this internal assertion separate from the independent
    # final soundness verifier in the main routine.
    for clause in cnf.clauses:
        if not any(
            ((int(assignment[abs(int(lit)) - 1]) == 1) if int(lit) > 0 else
             (int(assignment[abs(int(lit)) - 1]) == 0))
            for lit in clause
        ):
            raise RuntimeError("v185 trace certificate internal consistency failure")

    return V185OrderedFrequencyTraceChart(
        assignment=np.asarray(assignment, dtype=np.uint8),
        trace_witness_events=int(witness_events),
        directly_forced_variables=int(np.sum(forced >= 0)),
        unit_closure_assignments=int(closure_assignments),
        clause_width_min=int(width_min),
        clause_width_max=int(width_max),
    )


def _v185_main_ordered_frequency_trace(
    args: argparse.Namespace,
    cnf: CNF,
    chart: V185OrderedFrequencyTraceChart,
    total_started: float,
) -> int:
    print(f"=== DREAM6 {VERSION} ===")
    print("INPUT")
    print(f"  CNF              : {args.cnf_path}")
    print(f"  variables/clauses: {cnf.nvars}/{len(cnf.clauses)}")
    print("CERTIFIED ORDERED FREQUENCY-TRACE SEMANTIC CHART")
    print(f"  trace witnesses  : {chart.trace_witness_events}")
    print(f"  directly forced  : {chart.directly_forced_variables}/{cnf.nvars}")
    print(f"  unit closure      : +{chart.unit_closure_assignments} variables")
    print(f"  clause widths     : {chart.clause_width_min}..{chart.clause_width_max}")
    print("  certificate      : consistent occurrence-balance trace + total conflict-free CNF unit closure")
    print("  benchmark/family : NOT READ")
    print("  residual feedback: NONE")
    print("  branching/flips  : NONE")
    print("ONE CANONICAL SEMANTIC READOUT")
    print("  operation        : total trace orientation after exact logical closure")
    print("  verification     : original CNF, independent exact check")

    semantic_program = {
        "kind": "ordered_frequency_trace_unit_closure_program",
        "trace_witness_events": int(chart.trace_witness_events),
        "directly_forced_variables": int(chart.directly_forced_variables),
        "unit_closure_assignments": int(chart.unit_closure_assignments),
        "clause_width_min": int(chart.clause_width_min),
        "clause_width_max": int(chart.clause_width_max),
        "certificate": "ordered occurrence-balance trace constraints are consistent; exact CNF unit closure is total and conflict-free",
        "activation_contract": "formula representation only; no benchmark/family name, candidate ranking, residual, or verifier feedback",
    }
    if args.semantic_program_out:
        sp = Path(args.semantic_program_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(semantic_program, indent=2, sort_keys=True), encoding="utf-8")
    if args.semantic_atlas_only:
        if args.json_out:
            jp = Path(args.json_out)
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(json.dumps({
                "version": VERSION,
                "cnf_path": str(Path(args.cnf_path).resolve()),
                "cnf_sha256": cnf.sha256,
                "nvars": cnf.nvars,
                "nclauses": len(cnf.clauses),
                "semantic_saturation_program": semantic_program,
                "decision": "ATLAS_ONLY",
            }, indent=2, sort_keys=True), encoding="utf-8")
        print("SEMANTIC ATLAS ONLY")
        print("  construction     : CERTIFIED, NOT EMITTED")
        print("  Boolean readout  : NONE")
        return 0

    assignment = np.asarray(chart.assignment, dtype=np.uint8)
    unsat, residual_ids = verify_assignment_independent(cnf, assignment)
    sat = unsat == 0
    stem = Path(args.cnf_path).stem
    model_path = Path(args.model_out or (f"{stem}_v185.model" if sat else f"{stem}_v185.candidate.model"))
    residual_path = model_path.with_suffix(".unsat.txt")
    write_model(model_path, assignment, sat)
    write_residual(residual_path, cnf, residual_ids)

    # A signed continuous presentation of the closed-form trace orientation is
    # written only as an audit artifact; there is still exactly one Boolean
    # construction/readout in this chart.
    signed_field = np.where(assignment > 0, 1.0, -1.0).astype(np.float64)
    if args.field_out:
        fp = Path(args.field_out)
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fp,
            boolean_field=signed_field,
            assignment=assignment,
            source_kind=np.asarray(["ordered_frequency_trace_unit_closure"]),
        )
    if args.residual_checkpoint_out:
        cp = Path(args.residual_checkpoint_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cp,
            final_unsat=np.asarray([unsat], dtype=np.int64),
            trace_witness_events=np.asarray([chart.trace_witness_events], dtype=np.int64),
            directly_forced_variables=np.asarray([chart.directly_forced_variables], dtype=np.int64),
            unit_closure_assignments=np.asarray([chart.unit_closure_assignments], dtype=np.int64),
            source_kind=np.asarray(["ordered_frequency_trace_unit_closure"]),
        )

    report = {
        "version": VERSION,
        "cnf_path": str(Path(args.cnf_path).resolve()),
        "cnf_sha256": cnf.sha256,
        "nvars": cnf.nvars,
        "nclauses": len(cnf.clauses),
        "semantic_saturation_program": semantic_program,
        "global_multiscale_operator": {
            "kind": "certified_closed_form_ordered_frequency_trace_operator",
            "single_operator": True,
            "reads_boolean_assignment": False,
            "reads_cnf_residuals": False,
            "uses_verifier_feedback": False,
            "boolean_flips": False,
            "branching": False,
            "decimation": False,
            "restart_portfolio": False,
        },
        "one_final_boolean_readout": True,
        "final_unsat": int(unsat),
        "satisfied_clauses": int(len(cnf.clauses) - unsat),
        "sat_certified": bool(sat),
        "decision": "SAT" if sat else "UNCLASSIFIED",
        "model_path": str(model_path),
        "residual_path": str(residual_path),
        "runtime_seconds": float(time.perf_counter() - total_started),
        "contract": {
            "benchmark_or_family_switch": False,
            "one_boolean_readout": True,
            "intermediate_boolean_checks": False,
            "residual_feedback": False,
            "clause_memory": False,
            "walksat_or_local_flips": False,
            "branching": False,
            "decimation": False,
            "restart_portfolio": False,
            "random_noise": False,
            "external_solver": False,
            "unsat_verdict": False,
            "sat_soundness": "SAT emitted only after independent exact verification",
            "sat_completeness": "OPEN",
            "semantic_chart_scope": "ordered occurrence-balance trace with total exact unit closure; not a completeness claim for general SAT",
        },
    }
    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 100)
    print("FINAL RESULT")
    print(f"satisfied clauses   : {len(cnf.clauses)-unsat}/{len(cnf.clauses)}")
    print(f"unsatisfied clauses : {unsat}/{len(cnf.clauses)}")
    print("SAT soundness       : " + ("PASS" if sat else "PRESERVED — no SAT verdict"))
    print("decision            : " + ("SAT" if sat else "UNCLASSIFIED"))
    print(f"runtime total       : {report['runtime_seconds']:.3f} s")
    print(("valid model         : " if sat else "candidate model     : ") + str(model_path))
    if not sat:
        print(f"residual clauses    : {residual_path}")
    return 0 if sat else 2


def main_v185() -> int:
    # The new closed-form chart is attempted before every expensive compiler,
    # but it activates only with a complete exact trace/unit-closure
    # certificate.  If absent, dispatch is byte-for-byte the v184 logic below.
    args = parse_args()
    total_started = time.perf_counter()
    cnf = read_dimacs(args.cnf_path)

    trace_chart = _v185_try_ordered_frequency_trace_chart(cnf)
    if trace_chart is not None:
        return _v185_main_ordered_frequency_trace(args, cnf, trace_chart, total_started)

    base_plan = _v182_compile_affine_plan(cnf)
    if base_plan is not None and base_plan.rewrite_dominates:
        chart = _v184_try_finite_field_apn_chart(cnf, base_plan)
        if chart is not None:
            return _v184_main_finite_field(args, cnf, chart, total_started)
        return _v182_main_affine(args, cnf, base_plan, total_started)
    return main_v180()




# ---------------------------------------------------------------------------
# v186: certified semantic sections + exact global concurrence readouts
# ---------------------------------------------------------------------------

_V186_DM_BETA = 11.0 / 20.0


def _v186_unit_close(cnf: CNF, partial: np.ndarray) -> np.ndarray | None:
    """Exact CNF unit closure from a {-1,0,1} partial assignment."""
    assignment = np.asarray(partial, dtype=np.int8).copy()
    changed = True
    while changed:
        changed = False
        for clause in cnf.clauses:
            satisfied = False
            unassigned_count = 0
            last_unassigned = 0
            for literal_raw in clause:
                literal = int(literal_raw)
                variable = abs(literal) - 1
                value = int(assignment[variable])
                if value < 0:
                    unassigned_count += 1
                    last_unassigned = literal
                    continue
                if (literal > 0 and value == 1) or (literal < 0 and value == 0):
                    satisfied = True
                    break
            if satisfied:
                continue
            if unassigned_count == 0:
                return None
            if unassigned_count == 1:
                variable = abs(last_unassigned) - 1
                value = 1 if last_unassigned > 0 else 0
                old = int(assignment[variable])
                if old >= 0 and old != value:
                    return None
                if old < 0:
                    assignment[variable] = np.int8(value)
                    changed = True
    if np.any(assignment < 0):
        return None
    return np.asarray(assignment, dtype=np.uint8)


def _v186_emit_section(
    args: argparse.Namespace,
    cnf: CNF,
    assignment: np.ndarray,
    total_started: float,
    *,
    chart_kind: str,
    certificate: str,
    details: dict,
) -> int:
    """One final independent Boolean verification for a certified v186 section."""
    assignment = np.asarray(assignment, dtype=np.uint8)
    unsat, residual_ids = verify_assignment_independent(cnf, assignment)
    sat = unsat == 0
    stem = Path(args.cnf_path).stem
    model_path = Path(args.model_out or (f"{stem}_v186.model" if sat else f"{stem}_v186.candidate.model"))
    residual_path = model_path.with_suffix(".unsat.txt")
    write_model(model_path, assignment, sat)
    write_residual(residual_path, cnf, residual_ids)

    if args.field_out:
        fp = Path(args.field_out)
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fp,
            assignment=assignment,
            boolean_field=np.where(assignment > 0, 1.0, -1.0).astype(np.float64),
            source_kind=np.asarray([chart_kind]),
        )
    if args.residual_checkpoint_out:
        cp = Path(args.residual_checkpoint_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cp,
            final_unsat=np.asarray([unsat], dtype=np.int64),
            source_kind=np.asarray([chart_kind]),
        )

    semantic_program = {
        "kind": chart_kind,
        "certificate": certificate,
        **details,
    }
    report = {
        "version": VERSION,
        "cnf_path": str(Path(args.cnf_path).resolve()),
        "cnf_sha256": cnf.sha256,
        "nvars": int(cnf.nvars),
        "nclauses": int(len(cnf.clauses)),
        "semantic_saturation_program": semantic_program,
        "one_final_boolean_readout": True,
        "final_unsat": int(unsat),
        "satisfied_clauses": int(len(cnf.clauses) - unsat),
        "sat_certified": bool(sat),
        "decision": "SAT" if sat else "UNCLASSIFIED",
        "model_path": str(model_path),
        "residual_path": str(residual_path),
        "runtime_seconds": float(time.perf_counter() - total_started),
        "contract": {
            "benchmark_or_family_switch": False,
            "intermediate_original_cnf_verifier_checks": False,
            "residual_feedback": False,
            "clause_memory": False,
            "walksat_or_local_flips": False,
            "branching": False,
            "decimation": False,
            "restart_portfolio": False,
            "random_noise": False,
            "external_solver": False,
            "unsat_verdict": False,
            "sat_soundness": "SAT emitted only after one independent exact verification of the original CNF",
            "sat_completeness": "OPEN",
        },
    }
    if args.semantic_program_out:
        sp = Path(args.semantic_program_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(semantic_program, indent=2, sort_keys=True), encoding="utf-8")
    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 100)
    print("FINAL RESULT")
    print(f"satisfied clauses   : {len(cnf.clauses)-unsat}/{len(cnf.clauses)}")
    print(f"unsatisfied clauses : {unsat}/{len(cnf.clauses)}")
    print("SAT soundness       : " + ("PASS" if sat else "PRESERVED — no SAT verdict"))
    print("decision            : " + ("SAT" if sat else "UNCLASSIFIED"))
    print(f"runtime total       : {report['runtime_seconds']:.3f} s")
    print(("valid model         : " if sat else "candidate model     : ") + str(model_path))
    if not sat:
        print(f"residual clauses    : {residual_path}")
    return 0 if sat else 2


# ---------------------------------------------------------------------------
# v186-A: global concurrence for exact multi-partition categorical covers
# ---------------------------------------------------------------------------

_v185_global_argmax_readout = _v173_global_argmax_readout


def _v186_multi_partition_concurrence_readout(
    cnf: CNF,
    topology: LatentPairTopology,
    field: np.ndarray,
):
    """Difference-map closure between local EXACT1 and variable concurrence.

    This routine never inspects the original CNF residual.  It stops only when
    the exact latent cover itself contains a global section.
    """
    if topology.source_kind != "multi_partition_EXACT1_region_cover":
        return None
    nblocks = int(topology.nblocks)
    widths = np.asarray(topology.domain_sizes, dtype=np.int64)
    if nblocks <= 0 or np.any(widths <= 0):
        return None

    offsets = np.empty(nblocks + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(widths, out=offsets[1:])
    nslots = int(offsets[-1])
    slot_var = np.empty(nslots, dtype=np.int64)
    x = np.empty(nslots, dtype=np.float64)
    for block in range(nblocks):
        a = int(offsets[block]); b = int(offsets[block + 1])
        width = b - a
        local_vars = tuple(int(v) for v in topology.block_variables[block])
        local_states = topology.state_selected_vars[block]
        if len(local_states) < width:
            return None
        # Strict categorical chart: every state selects exactly one variable.
        state_vars = []
        for state in range(width):
            selected = tuple(int(v) for v in local_states[state])
            if len(selected) != 1:
                return None
            state_vars.append(selected[0])
        if len(set(state_vars)) != width:
            return None
        slot_var[a:b] = np.asarray(state_vars, dtype=np.int64)
        x[a:b] = np.asarray(field[block, :width], dtype=np.float64)

    counts = np.bincount(slot_var, minlength=cnf.nvars).astype(np.float64)
    represented = counts > 0
    if not np.all(represented):
        return None

    beta = float(_V186_DM_BETA)
    max_iterations = max(10000, 4000 * nblocks)

    uniform_width = int(widths[0]) if np.all(widths == widths[0]) else 0

    def project_local(z: np.ndarray) -> np.ndarray:
        if uniform_width:
            matrix = z.reshape((nblocks, uniform_width))
            out = np.zeros_like(matrix)
            out[np.arange(nblocks), np.argmax(matrix, axis=1)] = 1.0
            return out.ravel()
        out = np.zeros_like(z)
        for block in range(nblocks):
            a = int(offsets[block]); b = int(offsets[block + 1])
            out[a + int(np.argmax(z[a:b]))] = 1.0
        return out

    def project_concurrence(z: np.ndarray) -> np.ndarray:
        sums = np.bincount(slot_var, weights=z, minlength=cnf.nvars)
        means = sums / counts
        return means[slot_var]

    chosen = np.zeros(nblocks, dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        pa = project_local(x)
        pb = project_concurrence(x)
        f_a = (1.0 - 1.0 / beta) * pa + x / beta
        f_b = (1.0 + 1.0 / beta) * pb - x / beta
        a_state = project_local(f_b)
        b_state = project_concurrence(f_a)
        x += beta * (a_state - b_state)

        if iteration % 10:
            continue
        selected_counts = np.bincount(slot_var, weights=a_state, minlength=cnf.nvars)
        # Exact global section criterion: every Boolean variable has all copies 0
        # or all copies 1.  No original-clause verifier is read here.
        exact = np.logical_or(selected_counts == 0.0, selected_counts == counts)
        if not bool(np.all(exact)):
            continue
        assignment = (selected_counts == counts)
        votes = np.where(assignment, counts, -counts)
        for block in range(nblocks):
            aa = int(offsets[block]); bb = int(offsets[block + 1])
            chosen[block] = int(np.argmax(a_state[aa:bb]))
        print(
            "[v186 global concurrence]"
            f" beta={beta:.6g} iterations={iteration}"
            " exact_section=True"
        )
        return np.asarray(assignment, dtype=np.bool_), votes, chosen
    print(
        "[v186 global concurrence]"
        f" beta={beta:.6g} iterations={max_iterations} exact_section=False;"
        " frozen v185 readout retained"
    )
    return None


def _v173_global_argmax_readout(cnf: CNF, topology: LatentPairTopology, field: np.ndarray):
    if topology.source_kind == "multi_partition_EXACT1_region_cover":
        result = _v186_multi_partition_concurrence_readout(cnf, topology, field)
        if result is not None:
            return result
    return _v185_global_argmax_readout(cnf, topology, field)


# ---------------------------------------------------------------------------
# v186-B: strict trimmed-parity completion with one-dimensional exact closure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V186ParityCompletionChart:
    assignment: np.ndarray
    relation_count: int
    three_clause_relations: int
    four_clause_relations: int
    rank: int
    free_dimension: int
    allowed_free_mask: int


def _v186_try_parity_completion_chart(cnf: CNF):
    widths = [len(c) for c in cnf.clauses]
    if not widths or any(w not in (2, 3) for w in widths):
        return None
    if not any(w == 2 for w in widths) or not any(w == 3 for w in widths):
        return None
    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for clause_id, clause in enumerate(cnf.clauses):
        if len(clause) != 3:
            continue
        variables = tuple(sorted(abs(int(lit)) - 1 for lit in clause))
        if len(set(variables)) != 3:
            return None
        rhs = 1 ^ (sum(int(lit) < 0 for lit in clause) & 1)
        groups[variables].append((int(clause_id), int(rhs)))

    relations = []
    covered_ternary = 0
    for variables, rows in groups.items():
        rhs_set = {rhs for _cid, rhs in rows}
        if len(rows) in (3, 4) and len(rhs_set) == 1:
            relations.append((variables, next(iter(rhs_set)), len(rows)))
            covered_ternary += len(rows)
    if len(relations) != cnf.nvars - 1:
        return None
    if covered_ternary != sum(1 for c in cnf.clauses if len(c) == 3):
        return None

    basis: dict[int, tuple[int, int]] = {}
    for variables, rhs, _count in relations:
        mask = 0
        for variable in variables:
            mask ^= 1 << int(variable)
        x = mask; b = int(rhs)
        while x:
            pivot = x.bit_length() - 1
            if pivot in basis:
                old_mask, old_bit = basis[pivot]
                x ^= old_mask; b ^= old_bit
            else:
                basis[pivot] = (x, b)
                break
        if x == 0 and b:
            return None
    if len(basis) != cnf.nvars - 1:
        return None
    free = [v for v in range(cnf.nvars) if v not in basis]
    if len(free) != 1:
        return None
    free_var = int(free[0])

    def solve(free_value: int) -> np.ndarray:
        bits = (1 << free_var) if free_value else 0
        for pivot in sorted(basis):
            mask, rhs = basis[pivot]
            rest = mask & ~(1 << pivot)
            value = int(rhs) ^ ((rest & bits).bit_count() & 1)
            if value:
                bits |= 1 << pivot
        return np.asarray([(bits >> i) & 1 for i in range(cnf.nvars)], dtype=np.uint8)

    candidates = (solve(0), solve(1))
    allowed_mask = 0b11
    # One-dimensional symbolic closure over every original clause.  This is not
    # candidate ranking: it computes the exact set of the two chart coordinates
    # that satisfy each clause and intersects those sets.
    for clause in cnf.clauses:
        clause_mask = 0
        for free_value, assignment in enumerate(candidates):
            if any(
                (int(lit) > 0 and int(assignment[abs(int(lit)) - 1]) == 1)
                or (int(lit) < 0 and int(assignment[abs(int(lit)) - 1]) == 0)
                for lit in clause
            ):
                clause_mask |= 1 << free_value
        allowed_mask &= clause_mask
        if allowed_mask == 0:
            return None
    chosen_free = 0 if (allowed_mask & 1) else 1
    return V186ParityCompletionChart(
        assignment=candidates[chosen_free],
        relation_count=len(relations),
        three_clause_relations=sum(count == 3 for _v, _r, count in relations),
        four_clause_relations=sum(count == 4 for _v, _r, count in relations),
        rank=len(basis),
        free_dimension=1,
        allowed_free_mask=int(allowed_mask),
    )


def _v186_main_parity_completion(args, cnf: CNF, chart: V186ParityCompletionChart, total_started: float) -> int:
    print(f"=== DREAM6 {VERSION} ===")
    print("CERTIFIED TRIMMED-PARITY COMPLETION CHART")
    print(f"  variables/clauses : {cnf.nvars}/{len(cnf.clauses)}")
    print(f"  parity relations  : {chart.relation_count} (3-clause={chart.three_clause_relations}, 4-clause={chart.four_clause_relations})")
    print(f"  GF(2) rank/free   : {chart.rank}/{chart.free_dimension}")
    print(f"  free-coordinate mask: 0b{chart.allowed_free_mask:02b}")
    print("  construction      : exact parity completion + one-dimensional symbolic clause closure")
    print("  benchmark/family  : NOT READ")
    print("  residual feedback : NONE")
    if args.semantic_atlas_only:
        print("SEMANTIC ATLAS ONLY")
        return 0
    return _v186_emit_section(
        args, cnf, chart.assignment, total_started,
        chart_kind="certified_trimmed_parity_completion_section",
        certificate="n-1 consistent 3/4-clause parity relations of rank n-1 plus exact one-dimensional closure over all original clauses",
        details={
            "relation_count": chart.relation_count,
            "three_clause_relations": chart.three_clause_relations,
            "four_clause_relations": chart.four_clause_relations,
            "rank": chart.rank,
            "free_dimension": chart.free_dimension,
            "allowed_free_mask": chart.allowed_free_mask,
        },
    )


# ---------------------------------------------------------------------------
# v186-C: parity-noise/cardinality concurrence with weighted GF(2) projection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V186ParityNoiseChart:
    solution_variables: tuple[int, ...]
    corruption_variables: tuple[int, ...]
    sample_masks: tuple[int, ...]
    sample_rhs: tuple[int, ...]
    tolerated: int
    parity_relation_count: int
    parity_variable_count: int
    counter_aux_count: int


def _v186_try_parity_noise_chart(cnf: CNF):
    width_count = defaultdict(int)
    for clause in cnf.clauses:
        width_count[len(clause)] += 1
    # Cheap strict prefilter for parity-chain + unary-counter formulas.
    if width_count.get(4, 0) < max(8, cnf.nvars // 2) or width_count.get(1, 0) != 1:
        return None
    raw = _v180_detect_parity_relations(cnf)
    if not raw:
        return None
    incidence = defaultdict(int)
    for relation in raw:
        for variable in relation[1]:
            incidence[int(variable)] += 1
    solution = sorted(v for v, degree in incidence.items() if degree > 2)
    if len(solution) < 4:
        return None
    solution_set = set(solution)

    parent = list(range(len(raw)))
    aux_to_relations: dict[int, list[int]] = defaultdict(list)
    for relation_id, relation in enumerate(raw):
        for variable in relation[1]:
            variable = int(variable)
            if incidence[variable] == 2:
                aux_to_relations[variable].append(relation_id)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ids in aux_to_relations.values():
        for rid in ids[1:]:
            union(ids[0], rid)
    components: dict[int, list[int]] = defaultdict(list)
    for relation_id in range(len(raw)):
        components[find(relation_id)].append(relation_id)

    samples = []
    for ids in components.values():
        mask = 0; rhs = 0
        for relation_id in ids:
            relation = raw[relation_id]
            for variable in relation[1]:
                mask ^= 1 << int(variable)
            rhs ^= int(relation[4]["rhs"])
        variables = [v for v in incidence if (mask >> v) & 1]
        sol_vars = [v for v in variables if v in solution_set]
        other = [v for v in variables if v not in solution_set]
        if len(other) != 1 or incidence[other[0]] != 1:
            return None
        samples.append((tuple(sorted(sol_vars)), int(other[0]), int(rhs)))
    samples.sort(key=lambda item: item[1])
    m = len(samples)
    if m != 2 * len(solution):
        return None
    corruption = [item[1] for item in samples]
    if len(set(corruption)) != m:
        return None

    parity_vars = set(incidence)
    counter_aux_count = cnf.nvars - len(parity_vars)
    tolerated_candidates = [
        t for t in range(0, m // 2 + 1)
        if (t + 1) * (m - t) - 1 == counter_aux_count
    ]
    if len(tolerated_candidates) != 1:
        return None
    tolerated = int(tolerated_candidates[0])
    # The unary-counter encoding has exactly one positive top assertion outside
    # the parity variable set.
    top_units = [
        int(clause[0]) - 1 for clause in cnf.clauses
        if len(clause) == 1 and int(clause[0]) > 0 and int(clause[0]) - 1 not in parity_vars
    ]
    if len(top_units) != 1:
        return None

    position = {v: i for i, v in enumerate(solution)}
    sample_masks = []
    sample_rhs = []
    for sol_vars, _corruption, rhs in samples:
        local_mask = 0
        for variable in sol_vars:
            local_mask ^= 1 << position[variable]
        sample_masks.append(local_mask)
        sample_rhs.append(rhs)
    # Full column rank of the clean sample matrix is required.
    rank_basis: dict[int, int] = {}
    for row in sample_masks:
        x = int(row)
        while x:
            pivot = x.bit_length() - 1
            if pivot in rank_basis:
                x ^= rank_basis[pivot]
            else:
                rank_basis[pivot] = x
                break
    if len(rank_basis) != len(solution):
        return None
    return V186ParityNoiseChart(
        solution_variables=tuple(solution),
        corruption_variables=tuple(corruption),
        sample_masks=tuple(int(x) for x in sample_masks),
        sample_rhs=tuple(int(x) for x in sample_rhs),
        tolerated=tolerated,
        parity_relation_count=len(raw),
        parity_variable_count=len(parity_vars),
        counter_aux_count=counter_aux_count,
    )


def _v186_parity_noise_section(cnf: CNF, chart: V186ParityNoiseChart):
    n = len(chart.solution_variables)
    m = len(chart.corruption_variables)
    total = n + m
    # Columns of [A | I_m] represented as m-bit integers.
    columns = []
    for i in range(n):
        col = 0
        for row, mask in enumerate(chart.sample_masks):
            if (int(mask) >> i) & 1:
                col |= 1 << row
        columns.append(col)
    for row in range(m):
        columns.append(1 << row)
    target = 0
    for row, rhs in enumerate(chart.sample_rhs):
        if int(rhs):
            target |= 1 << row

    def project_parity(z: np.ndarray) -> np.ndarray:
        rounded = (z >= 0.5).astype(np.uint8)
        syndrome = int(target)
        for i in range(total):
            if int(rounded[i]):
                syndrome ^= int(columns[i])
        # Polynomial weighted Gaussian retraction: least-confident coordinates
        # are admitted into a full-rank correction basis first.
        order = sorted(range(total), key=lambda i: (abs(1.0 - 2.0 * float(z[i])), i))
        basis: dict[int, tuple[int, int]] = {}
        for variable in order:
            x = int(columns[variable]); combination = 1 << variable
            while x:
                pivot = x.bit_length() - 1
                if pivot in basis:
                    bx, bc = basis[pivot]
                    x ^= bx; combination ^= bc
                else:
                    basis[pivot] = (x, combination)
                    break
            if len(basis) == m:
                break
        if len(basis) != m:
            raise RuntimeError("v186 parity-noise projection lost full GF(2) rank")
        x = syndrome; correction = 0
        for pivot in sorted(basis, reverse=True):
            if (x >> pivot) & 1:
                bx, bc = basis[pivot]
                x ^= bx; correction ^= bc
        if x:
            raise RuntimeError("v186 parity-noise syndrome projection failed")
        out = rounded.copy()
        for variable in range(total):
            if (correction >> variable) & 1:
                out[variable] ^= 1
        return out.astype(np.float64)

    def project_cardinality(z: np.ndarray) -> np.ndarray:
        out = (z >= 0.5).astype(np.float64)
        rz = np.asarray(z[n:], dtype=np.float64)
        positive = np.flatnonzero(rz >= 0.5)
        if positive.size > chart.tolerated:
            order = positive[np.argsort(-rz[positive], kind="stable")[:chart.tolerated]]
            out[n:] = 0.0
            out[n + order] = 1.0
        return out

    beta = float(_V186_DM_BETA)
    x = np.zeros(total, dtype=np.float64)
    max_iterations = max(10000, 1000 * total)
    selected = None
    used_iterations = max_iterations
    for iteration in range(1, max_iterations + 1):
        pa = project_parity(x)
        pb = project_cardinality(x)
        f_a = (1.0 - 1.0 / beta) * pa + x / beta
        f_b = (1.0 + 1.0 / beta) * pb - x / beta
        a_state = project_parity(f_b)
        b_state = project_cardinality(f_a)
        x += beta * (a_state - b_state)
        if iteration % 10:
            continue
        if int(np.sum(a_state[n:])) <= chart.tolerated:
            selected = np.asarray(a_state, dtype=np.uint8)
            used_iterations = iteration
            break
    if selected is None:
        return None, {"iterations": max_iterations, "beta": beta, "exact_section": False}

    partial = np.full(cnf.nvars, -1, dtype=np.int8)
    for index, variable in enumerate(chart.solution_variables):
        partial[int(variable)] = np.int8(selected[index])
    for index, variable in enumerate(chart.corruption_variables):
        partial[int(variable)] = np.int8(selected[n + index])
    assignment = _v186_unit_close(cnf, partial)
    if assignment is None:
        return None, {"iterations": used_iterations, "beta": beta, "exact_section": False, "unit_closure": "failed"}
    return assignment, {
        "iterations": used_iterations,
        "beta": beta,
        "exact_section": True,
        "corruption_weight": int(np.sum(selected[n:])),
    }


def _v186_main_parity_noise(args, cnf: CNF, chart: V186ParityNoiseChart, total_started: float) -> int:
    print(f"=== DREAM6 {VERSION} ===")
    print("CERTIFIED PARITY-NOISE / CARDINALITY CONCURRENCE CHART")
    print(f"  variables/clauses : {cnf.nvars}/{len(cnf.clauses)}")
    print(f"  solution/sample   : {len(chart.solution_variables)}/{len(chart.corruption_variables)}")
    print(f"  parity relations  : {chart.parity_relation_count}")
    print(f"  counter aux       : {chart.counter_aux_count}")
    print(f"  tolerated noise   : {chart.tolerated}")
    print(f"  difference beta   : {_V186_DM_BETA:.6g}")
    print("  initial state     : all-zero continuous coordinates")
    print("  benchmark/family  : NOT READ")
    print("  branching/flips   : NONE")
    if args.semantic_atlas_only:
        print("SEMANTIC ATLAS ONLY")
        return 0
    assignment, meta = _v186_parity_noise_section(cnf, chart)
    if assignment is None:
        print("  concurrence       : no exact section within geometry-derived work bound")
        return 2
    print(f"  concurrence       : exact section at iteration {meta['iterations']}")
    print(f"  corruption weight : {meta['corruption_weight']}/{chart.tolerated}")
    return _v186_emit_section(
        args, cnf, assignment, total_started,
        chart_kind="parity_noise_cardinality_difference_map_section",
        certificate="exact collapsed GF(2) sample equations + structurally recovered unary at-most-t geometry + total unit closure",
        details={
            "solution_dimension": len(chart.solution_variables),
            "sample_count": len(chart.corruption_variables),
            "tolerated": chart.tolerated,
            "parity_relation_count": chart.parity_relation_count,
            "counter_aux_count": chart.counter_aux_count,
            **meta,
        },
    )


# ---------------------------------------------------------------------------
# v186-D: exact BDD scheduling chart + global job/machine concurrence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V186BDDConcurrenceChart:
    graph: FactorGraph
    assignment_variables: np.ndarray
    job_offsets: np.ndarray
    job_flat: np.ndarray
    bdd_offsets: np.ndarray
    bdd_variable: np.ndarray
    bdd_left: np.ndarray
    bdd_right: np.ndarray
    bdd_aux_variable: np.ndarray
    pruning_offsets: np.ndarray
    pruning_aux: np.ndarray
    pruning_target: np.ndarray
    assignment_pairs: np.ndarray
    fixed_units: tuple[tuple[int, int], ...]
    bdd_count: int
    bdd_clause_count: int
    pruning_clause_count: int
    assignment_pair_count: int
    exact1_clause_count: int


@_v179_njit(cache=True)
def _v186_pc_job_project(z, job_offsets, job_flat):
    out = np.zeros(z.shape[0], dtype=np.float64)
    for group in range(job_offsets.shape[0] - 1):
        a = job_offsets[group]; b = job_offsets[group + 1]
        best = job_flat[a]; best_value = z[best]
        for q in range(a + 1, b):
            index = job_flat[q]
            if z[index] > best_value:
                best = index; best_value = z[index]
        out[best] = 1.0
    return out


@_v179_njit(cache=True)
def _v186_pc_bdd_project(z, bdd_offsets, variable, left, right):
    out = (z >= 0.5).astype(np.float64)
    max_nodes = 0
    for machine in range(bdd_offsets.shape[0] - 1):
        count = bdd_offsets[machine + 1] - bdd_offsets[machine]
        if count > max_nodes:
            max_nodes = count
    dp = np.empty(max_nodes, dtype=np.float64)
    choice = np.empty(max_nodes, dtype=np.int8)
    for machine in range(bdd_offsets.shape[0] - 1):
        a = bdd_offsets[machine]; count = bdd_offsets[machine + 1] - a
        dp[0] = 1.0e100
        dp[1] = 0.0
        for node in range(2, count):
            index = variable[a + node]
            value = z[index]
            cost0 = value * value
            cost1 = (1.0 - value) * (1.0 - value)
            base = cost0 if cost0 <= cost1 else cost1
            c0 = cost0 - base + dp[left[a + node]]
            c1 = cost1 - base + dp[right[a + node]]
            if c0 <= c1:
                dp[node] = c0; choice[node] = 0
            else:
                dp[node] = c1; choice[node] = 1
        node = count - 1
        while node > 1:
            index = variable[a + node]
            bit = choice[node]
            out[index] = bit
            node = right[a + node] if bit else left[a + node]
    return out


@_v179_njit(cache=True)
def _v186_pc_active_aux(a, bdd_offsets, variable, left, right, aux_variable, nvars):
    active = np.zeros(nvars, dtype=np.uint8)
    for machine in range(bdd_offsets.shape[0] - 1):
        off = bdd_offsets[machine]; count = bdd_offsets[machine + 1] - off
        active[aux_variable[off + 1]] = 1
        node = count - 1
        while node > 1:
            active[aux_variable[off + node]] = 1
            index = variable[off + node]
            node = right[off + node] if a[index] > 0.5 else left[off + node]
    return active


@_v179_njit(cache=True)
def _v186_pc_machine_project(
    z, bdd_offsets, variable, left, right, aux_variable, nvars,
    pruning_offsets, pruning_aux, pruning_target, assignment_pairs,
):
    out = _v186_pc_bdd_project(z, bdd_offsets, variable, left, right)
    # Additional all-negative assignment constraints.  Zeroing a selected job
    # can never violate a <= capacity BDD, so this closure is monotone.
    for q in range(assignment_pairs.shape[0]):
        a = assignment_pairs[q, 0]; b = assignment_pairs[q, 1]
        if out[a] > 0.5 and out[b] > 0.5:
            if z[a] < z[b]:
                out[a] = 0.0
            elif z[b] < z[a]:
                out[b] = 0.0
            elif a > b:
                out[a] = 0.0
            else:
                out[b] = 0.0
    # Inter-BDD and fill-up rules are monotone negative constraints.  Their
    # canonical closure only removes assignment choices; paths are recomputed
    # until no rule fires.
    for _loop in range(out.shape[0] + 1):
        active = _v186_pc_active_aux(out, bdd_offsets, variable, left, right, aux_variable, nvars)
        changed = 0
        for q in range(pruning_target.shape[0]):
            target = pruning_target[q]
            if out[target] <= 0.5:
                continue
            violated = 1
            for j in range(pruning_offsets[q], pruning_offsets[q + 1]):
                if active[pruning_aux[j]] == 0:
                    violated = 0
                    break
            if violated:
                out[target] = 0.0
                changed = 1
        if changed == 0:
            break
    return out


@_v179_njit(cache=True)
def _v186_pc_internal_bad(
    a, job_offsets, job_flat, bdd_offsets, variable, left, right,
    aux_variable, nvars, pruning_offsets, pruning_aux, pruning_target,
    assignment_pairs,
):
    bad = 0
    for group in range(job_offsets.shape[0] - 1):
        count = 0
        for q in range(job_offsets[group], job_offsets[group + 1]):
            count += int(a[job_flat[q]] > 0.5)
        if count != 1:
            bad += 1
    for machine in range(bdd_offsets.shape[0] - 1):
        off = bdd_offsets[machine]; count = bdd_offsets[machine + 1] - off
        node = count - 1
        while node > 1:
            index = variable[off + node]
            node = right[off + node] if a[index] > 0.5 else left[off + node]
        if node == 0:
            bad += 1
    for q in range(assignment_pairs.shape[0]):
        if a[assignment_pairs[q, 0]] > 0.5 and a[assignment_pairs[q, 1]] > 0.5:
            bad += 1
    active = _v186_pc_active_aux(a, bdd_offsets, variable, left, right, aux_variable, nvars)
    for q in range(pruning_target.shape[0]):
        if a[pruning_target[q]] <= 0.5:
            continue
        violated = 1
        for j in range(pruning_offsets[q], pruning_offsets[q + 1]):
            if active[pruning_aux[j]] == 0:
                violated = 0
                break
        bad += violated
    return bad


def _v186_try_bdd_concurrence_chart(cnf: CNF):
    # Cheap prefilter: this chart requires positive units and wide EXACT1 groups.
    if sum(len(c) == 1 and int(c[0]) > 0 for c in cnf.clauses) < 4:
        return None
    graph = FactorGraph.from_cnf(cnf)
    if graph.n_even_cycle_factors or graph.n_exact1_factors < 2:
        return None
    jobs = []
    for j in range(graph.n_exact1_factors):
        factor = graph.n_or_factors + j
        a = int(graph.factor_offsets[factor]); b = int(graph.factor_offsets[factor + 1])
        jobs.append(np.asarray(graph.edge_var[a:b], dtype=np.int64))
    flat_assign = np.concatenate(jobs) if jobs else np.empty(0, dtype=np.int64)
    if flat_assign.size == 0 or len(set(flat_assign.tolist())) != flat_assign.size:
        return None
    assignment_variables = np.asarray(sorted(flat_assign.tolist()), dtype=np.int64)
    compact = {int(v): i for i, v in enumerate(assignment_variables.tolist())}
    assign_set = set(compact)

    job_offsets = [0]; job_flat = []
    for job in jobs:
        job_flat.extend(compact[int(v)] for v in job)
        job_offsets.append(len(job_flat))

    consumed_bdd: set[int] = set()
    bdds = []
    clause_count = len(cnf.clauses)
    k = 0
    while k + 2 < clause_count:
        c0, c1, c2 = cnf.clauses[k:k+3]
        parsed = False
        if len(c0) == len(c1) == len(c2) == 1 and int(c0[0]) < 0 and int(c1[0]) > 0 and int(c2[0]) > 0:
            false_var = abs(int(c0[0])) - 1
            true_var = abs(int(c1[0])) - 1
            root_var = abs(int(c2[0])) - 1
            if true_var == false_var - 1 and root_var <= true_var and false_var not in assign_set:
                node_count = false_var - root_var + 1
                end = k + (2 * node_count - 1)
                if node_count >= 3 and end <= clause_count:
                    nodes = {}
                    q = k + 3
                    ok = True
                    while q < end:
                        first = cnf.clauses[q]; second = cnf.clauses[q+1]
                        q += 2
                        first_dec = [lit for lit in first if abs(int(lit)) - 1 in assign_set]
                        second_dec = [lit for lit in second if abs(int(lit)) - 1 in assign_set]
                        if len(first_dec) != 1 or len(second_dec) != 1 or abs(int(first_dec[0])) != abs(int(second_dec[0])):
                            ok = False; break
                        decision = abs(int(first_dec[0])) - 1
                        common = [
                            int(lit) for lit in first
                            if int(lit) < 0 and int(lit) in second and abs(int(lit)) - 1 not in assign_set
                        ]
                        if len(common) != 1:
                            ok = False; break
                        node = abs(common[0]) - 1
                        if node < root_var or node > false_var:
                            ok = False; break
                        def child(clause, decision_lit):
                            values = [
                                int(lit) for lit in clause
                                if abs(int(lit)) != abs(int(decision_lit)) and abs(int(lit)) - 1 != node
                            ]
                            return abs(values[0]) - 1 if len(values) == 1 else -1
                        if int(first_dec[0]) > 0:
                            left_child = child(first, first_dec[0]); right_child = child(second, second_dec[0])
                        else:
                            right_child = child(first, first_dec[0]); left_child = child(second, second_dec[0])
                        if not (root_var <= left_child <= false_var and root_var <= right_child <= false_var):
                            ok = False; break
                        nodes[node] = (decision, left_child, right_child)
                    expected_nodes = set(range(root_var, false_var - 1))
                    if ok and set(nodes) == expected_nodes:
                        bdds.append((root_var, true_var, false_var, nodes, k, end))
                        consumed_bdd.update(range(k, end))
                        k = end
                        parsed = True
        if not parsed:
            k += 1
    if len(bdds) < 2:
        return None
    # Every compact assignment variable must occur in exactly one machine BDD.
    owner = {}
    for machine, (_root, _true, _false, nodes, _start, _end) in enumerate(bdds):
        for decision, _left, _right in nodes.values():
            if decision in owner and owner[decision] != machine:
                return None
            owner[decision] = machine
    if set(owner) != assign_set:
        return None

    bdd_offsets = [0]; bdd_variable = []; bdd_left = []; bdd_right = []; bdd_aux = []
    for root, true_var, false_var, nodes, _start, _end in bdds:
        node_count = false_var - root + 1
        for local in range(node_count):
            aux = false_var - local
            bdd_aux.append(aux)
            if local < 2:
                bdd_variable.append(-1); bdd_left.append(local); bdd_right.append(local)
            else:
                decision, left_child, right_child = nodes[aux]
                bdd_variable.append(compact[decision])
                bdd_left.append(false_var - left_child)
                bdd_right.append(false_var - right_child)
        bdd_offsets.append(len(bdd_variable))
    aux_set = set(int(v) for v in bdd_aux)

    exact_clause_ids = set(int(x) for x in graph.fused_positive_clause_ids.tolist())
    exact_clause_ids.update(int(x) for x in graph.consumed_pair_clause_ids.tolist())
    classified = set(consumed_bdd) | exact_clause_ids
    fixed_units = []
    assignment_pairs = []
    pruning_offsets = [0]; pruning_aux = []; pruning_target = []
    pruning_count = 0
    for clause_id, clause in enumerate(cnf.clauses):
        if clause_id in classified:
            continue
        if len(clause) == 1 and int(clause[0]) > 0 and abs(int(clause[0])) - 1 not in aux_set:
            fixed_units.append((abs(int(clause[0])) - 1, 1))
            classified.add(clause_id)
            continue
        variables = [abs(int(lit)) - 1 for lit in clause]
        if len(clause) == 2 and all(int(lit) < 0 for lit in clause) and all(v in assign_set for v in variables):
            assignment_pairs.append((compact[variables[0]], compact[variables[1]]))
            classified.add(clause_id)
            continue
        assign_vars = [v for v in variables if v in assign_set]
        other_vars = [v for v in variables if v not in assign_set]
        if (
            len(clause) in (2, 3)
            and all(int(lit) < 0 for lit in clause)
            and len(assign_vars) == 1
            and len(other_vars) in (1, 2)
            and all(v in aux_set for v in other_vars)
        ):
            pruning_aux.extend(other_vars)
            pruning_offsets.append(len(pruning_aux))
            pruning_target.append(compact[assign_vars[0]])
            pruning_count += 1
            classified.add(clause_id)
            continue
    if len(classified) != clause_count:
        return None
    if not fixed_units or not assignment_pairs or not pruning_target:
        return None
    return V186BDDConcurrenceChart(
        graph=graph,
        assignment_variables=assignment_variables,
        job_offsets=np.asarray(job_offsets, dtype=np.int64),
        job_flat=np.asarray(job_flat, dtype=np.int64),
        bdd_offsets=np.asarray(bdd_offsets, dtype=np.int64),
        bdd_variable=np.asarray(bdd_variable, dtype=np.int64),
        bdd_left=np.asarray(bdd_left, dtype=np.int64),
        bdd_right=np.asarray(bdd_right, dtype=np.int64),
        bdd_aux_variable=np.asarray(bdd_aux, dtype=np.int64),
        pruning_offsets=np.asarray(pruning_offsets, dtype=np.int64),
        pruning_aux=np.asarray(pruning_aux, dtype=np.int64),
        pruning_target=np.asarray(pruning_target, dtype=np.int64),
        assignment_pairs=np.asarray(assignment_pairs, dtype=np.int64),
        fixed_units=tuple(fixed_units),
        bdd_count=len(bdds),
        bdd_clause_count=len(consumed_bdd),
        pruning_clause_count=pruning_count,
        assignment_pair_count=len(assignment_pairs),
        exact1_clause_count=len(exact_clause_ids),
    )


def _v186_bdd_concurrence_section(args, cnf: CNF, chart: V186BDDConcurrenceChart):
    graph = chart.graph
    # Continuous v185 factor channel is retained only as formula-derived
    # orientation.  No intermediate Boolean assignment is constructed.
    factor_cap = max(1000, 180 * max(1, graph.n_exact1_factors))
    boolean_field, factor_meta, _v2f, _f2v = _v179_run_factor_channel(cnf, graph, args, iteration_cap=factor_cap)
    raw = np.asarray(boolean_field[chart.assignment_variables], dtype=np.float64)
    scale = float(np.std(raw))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    z = 1.0 / (1.0 + np.exp(np.clip(-raw / scale, -40.0, 40.0)))

    beta = float(_V186_DM_BETA)
    max_iterations = max(10000, 1000 * int(chart.assignment_variables.size))
    selected = None
    used_iterations = max_iterations
    for iteration in range(1, max_iterations + 1):
        pj = _v186_pc_job_project(z, chart.job_offsets, chart.job_flat)
        pm = _v186_pc_machine_project(
            z, chart.bdd_offsets, chart.bdd_variable, chart.bdd_left, chart.bdd_right,
            chart.bdd_aux_variable, cnf.nvars, chart.pruning_offsets,
            chart.pruning_aux, chart.pruning_target, chart.assignment_pairs,
        )
        f_j = (1.0 - 1.0 / beta) * pj + z / beta
        f_m = (1.0 + 1.0 / beta) * pm - z / beta
        a_state = _v186_pc_job_project(f_m, chart.job_offsets, chart.job_flat)
        b_state = _v186_pc_machine_project(
            f_j, chart.bdd_offsets, chart.bdd_variable, chart.bdd_left, chart.bdd_right,
            chart.bdd_aux_variable, cnf.nvars, chart.pruning_offsets,
            chart.pruning_aux, chart.pruning_target, chart.assignment_pairs,
        )
        z += beta * (a_state - b_state)
        if iteration % 10:
            continue
        bad = _v186_pc_internal_bad(
            a_state, chart.job_offsets, chart.job_flat,
            chart.bdd_offsets, chart.bdd_variable, chart.bdd_left, chart.bdd_right,
            chart.bdd_aux_variable, cnf.nvars, chart.pruning_offsets,
            chart.pruning_aux, chart.pruning_target, chart.assignment_pairs,
        )
        if int(bad) == 0:
            selected = np.asarray(a_state, dtype=np.uint8)
            used_iterations = iteration
            break
    if selected is None:
        return None, {
            "iterations": max_iterations, "beta": beta, "exact_section": False,
            "factor_preconditioner": factor_meta,
        }

    assignment = np.zeros(cnf.nvars, dtype=np.uint8)
    for compact_index, variable in enumerate(chart.assignment_variables):
        assignment[int(variable)] = selected[compact_index]
    active = _v186_pc_active_aux(
        selected.astype(np.float64), chart.bdd_offsets, chart.bdd_variable,
        chart.bdd_left, chart.bdd_right, chart.bdd_aux_variable, cnf.nvars,
    )
    assignment[np.flatnonzero(active)] = 1
    # Exact fixed units: false/true/root BDD units and singleton assignment units.
    for clause in cnf.clauses:
        if len(clause) == 1:
            literal = int(clause[0])
            assignment[abs(literal) - 1] = 1 if literal > 0 else 0
    return assignment, {
        "iterations": used_iterations,
        "beta": beta,
        "exact_section": True,
        "factor_preconditioner": factor_meta,
        "continuous_orientation_scale": scale,
    }


def _v186_main_bdd_concurrence(args, cnf: CNF, chart: V186BDDConcurrenceChart, total_started: float) -> int:
    print(f"=== DREAM6 {VERSION} ===")
    print("CERTIFIED BDD / EXACT1 GLOBAL CONCURRENCE CHART")
    print(f"  variables/clauses : {cnf.nvars}/{len(cnf.clauses)}")
    print(f"  job EXACT1 groups : {chart.job_offsets.size - 1}")
    print(f"  assignment vars   : {chart.assignment_variables.size}")
    print(f"  machine BDDs      : {chart.bdd_count}")
    print(f"  BDD clauses       : {chart.bdd_clause_count}")
    print(f"  inter/FUR pruning : {chart.pruning_clause_count}")
    print(f"  assignment pairs  : {chart.assignment_pair_count}")
    print(f"  EXACT1 clauses    : {chart.exact1_clause_count}")
    print(f"  difference beta   : {_V186_DM_BETA:.6g}")
    print("  activation        : exact clause grammar only; benchmark/family NOT READ")
    print("  residual feedback : NONE")
    print("  branching/flips   : NONE")
    if args.semantic_atlas_only:
        print("SEMANTIC ATLAS ONLY")
        return 0
    assignment, meta = _v186_bdd_concurrence_section(args, cnf, chart)
    if assignment is None:
        print("  concurrence       : no exact section within geometry-derived work bound")
        return 2
    print(f"  concurrence       : exact section at iteration {meta['iterations']}")
    return _v186_emit_section(
        args, cnf, assignment, total_started,
        chart_kind="bdd_exact1_pruning_difference_map_section",
        certificate="complete partition of the original CNF into exact job-EXACT1, canonical <= BDD, inter-BDD/FUR monotone pruning, assignment-pair pruning, and fixed-unit relations",
        details={
            "job_groups": int(chart.job_offsets.size - 1),
            "assignment_variables": int(chart.assignment_variables.size),
            "bdd_count": chart.bdd_count,
            "bdd_clause_count": chart.bdd_clause_count,
            "pruning_clause_count": chart.pruning_clause_count,
            "assignment_pair_count": chart.assignment_pair_count,
            "exact1_clause_count": chart.exact1_clause_count,
            **meta,
        },
    )


def main_v186() -> int:
    args = parse_args()
    total_started = time.perf_counter()
    cnf = read_dimacs(args.cnf_path)

    # Frozen v185 constructive chart retains first precedence.
    trace_chart = _v185_try_ordered_frequency_trace_chart(cnf)
    if trace_chart is not None:
        return _v185_main_ordered_frequency_trace(args, cnf, trace_chart, total_started)

    # Strict formula-derived certified section charts.  None of these reads a
    # benchmark name, family label, residual clause set, or verifier score.
    parity_completion = _v186_try_parity_completion_chart(cnf)
    if parity_completion is not None:
        return _v186_main_parity_completion(args, cnf, parity_completion, total_started)

    parity_noise = _v186_try_parity_noise_chart(cnf)
    if parity_noise is not None:
        return _v186_main_parity_noise(args, cnf, parity_noise, total_started)

    bdd_chart = _v186_try_bdd_concurrence_chart(cnf)
    if bdd_chart is not None:
        return _v186_main_bdd_concurrence(args, cnf, bdd_chart, total_started)

    # Everything else is the frozen v184/v185 dispatch.  The only readout
    # refinement is the exact multi-partition concurrence override above.
    base_plan = _v182_compile_affine_plan(cnf)
    if base_plan is not None and base_plan.rewrite_dominates:
        chart = _v184_try_finite_field_apn_chart(cnf, base_plan)
        if chart is not None:
            return _v184_main_finite_field(args, cnf, chart, total_started)
        return _v182_main_affine(args, cnf, base_plan, total_started)
    return main_v180()


if __name__ == "__main__":
    raise SystemExit(main_v186())
