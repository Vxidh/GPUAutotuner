"""Experimental designs (contract: screening, response_surface).

Coding: x_f = (log2(level) - C_f) / H_f with C, H from the screening low/high
levels, so the screening corners sit at exactly +-1.
"""
import itertools
import math

import numpy as np
from pyDOE3 import fracfact

from design_space import FACTORS, FULL_LEVELS, SCREEN_LEVELS, is_valid

C = {f: (math.log2(SCREEN_LEVELS[f][0]) + math.log2(SCREEN_LEVELS[f][1])) / 2 for f in FACTORS}
H = {f: (math.log2(SCREEN_LEVELS[f][1]) - math.log2(SCREEN_LEVELS[f][0])) / 2 for f in FACTORS}

GENERATOR = "a b c d e abcde"  # F = ABCDE, I = ABCDEF, resolution VI


def code(f, level):
    return (math.log2(level) - C[f]) / H[f]


def snap(f, log2_value):
    """Nearest legal power-of-two level; ties resolve to the lower level."""
    return min(FULL_LEVELS[f], key=lambda a: (abs(math.log2(a) - log2_value), a))


def screening_points():
    coded = fracfact(GENERATOR)
    pts = []
    for i, row in enumerate(coded, 1):
        cfg = {f: SCREEN_LEVELS[f][1] if row[j] > 0 else SCREEN_LEVELS[f][0]
               for j, f in enumerate(FACTORS)}
        assert is_valid(cfg)[0], cfg
        pts.append((f"S{i:02d}", cfg))
    return pts


def alias_table():
    """Alias pairs implied by I = ABCDEF for every main effect and 2FI."""
    letters = dict(zip("ABCDEF", FACTORS))
    word = set("ABCDEF")
    rows = []
    for r in (1, 2):
        for eff in itertools.combinations("ABCDEF", r):
            alias = "".join(sorted(word.symmetric_difference(eff)))
            rows.append({"effect": ":".join(letters[e] for e in eff),
                         "aliased_with": ":".join(letters[a] for a in alias),
                         "alias_order": len(alias)})
    return rows


def ccd_points(retained, fixed, n_center=4):
    """CCD on the retained factors. Core = screening design projected onto the
    retained factors (half-fraction if all 6, full 2^k otherwise)."""
    k = len(retained)
    if k == 6:
        core = [[1 if v > 0 else -1 for v in row] for row in fracfact(GENERATOR)]
        core = [dict(zip(FACTORS, r)) for r in core]
    else:
        core = [dict(zip(retained, s)) for s in itertools.product([-1, 1], repeat=k)]
    alpha = len(core) ** 0.25
    nominal = list(core)
    for f in retained:
        for s in (-1, 1):
            nominal.append({g: (s * alpha if g == f else 0.0) for g in retained})
    nominal += [{g: 0.0 for g in retained} for _ in range(n_center)]

    pts = []
    for i, p in enumerate(nominal, 1):
        cfg = {f: snap(f, C[f] + p[f] * H[f]) for f in retained}
        cfg.update(fixed)
        cfg = {f: cfg[f] for f in FACTORS}
        ok, reason = is_valid(cfg)
        assert ok, f"CCD point infeasible: {cfg} ({reason})"
        kind = "factorial" if i <= len(core) else ("axial" if i <= len(core) + 2 * k else "center")
        pts.append((f"C{i:02d}", cfg, {f: p[f] for f in retained}, kind))
    return pts, alpha
