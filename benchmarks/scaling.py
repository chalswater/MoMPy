#!/usr/bin/env python3
"""Reproduce the moment-matrix build benchmarks of the MoMPy tutorial.
 
This script rebuilds, from scratch, the eight scenario configurations reported
in Table III ("Build times for the moment matrix") of
 
    C. Roch i Carceller, "MoMPy: a declarative framework for moment matrices
    in semidefinite programming hierarchies".
 
Each scenario is a *configuration only*: a set of operator families, the
relations declared between them, and a monomial list.  Nothing here solves an
SDP -- what is being timed is exclusively ``MomentProblem.build()``, i.e. the
word-closure and variable-assignment step that the paper is about.  CVXPY is
therefore not needed to run this file.
 
Three of the four reported columns are deterministic functions of the
configuration and must reproduce exactly on any machine:
 
    matrix  the side length n of the built matrix, = len(monomials) + 1
    vars    the number of distinct SDP variables after quotienting
    words   the number of distinct words reached during closure
 
The fourth, the build time, is hardware- and version-dependent and will not
reproduce exactly.  Re-run this script on the machine you intend to quote and
use ``--latex`` to emit table rows, and ``--env`` to print the hardware and
version details that belong in the caption.
 
Usage
-----
    python benchmarks/scaling.py                    # all scenarios
    python benchmarks/scaling.py --fast             # skip the two slow ones
    python benchmarks/scaling.py -s npa_bipartite   # one scenario by key
    python benchmarks/scaling.py --repeat 3         # best-of-3 timing
    python benchmarks/scaling.py --latex            # emit LaTeX table rows
    python benchmarks/scaling.py --check            # compare against the paper
    python benchmarks/scaling.py --env              # environment report only
 
The reference values in ``PAPER`` below are the ones printed in the paper.
``--check`` compares the deterministic columns against them and exits non-zero
on any mismatch, which makes this file usable as a regression test on the
quotient construction itself.
"""
 
from __future__ import annotations
 
import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Callable
 
from MoMPy import Algebra, MomentProblem, OperatorSet
 
# --------------------------------------------------------------------------
# Reference values as printed in Table III of the paper.  matrix/vars/words are
# deterministic; seconds are indicative only (see the module docstring).
# --------------------------------------------------------------------------
PAPER = {
    "npa_bipartite":     dict(matrix=256,  vars=19_337,  words=324_931,    seconds=1.461),
    "pam_plain":         dict(matrix=3165, vars=27_071,  words=11_122_245, seconds=98.044),
    "pam_comm_states":   dict(matrix=3015, vars=12_785,  words=8_936_655,  seconds=82.191),
    "npa_tripartite":    dict(matrix=817,  vars=143_442, words=3_639_601,  seconds=23.374),
    "npa_hybrid":        dict(matrix=756,  vars=175_478, words=3_058_581,  seconds=17.418),
    "pam_povm_prep":     dict(matrix=1629, vars=428_683, words=3_591_037,  seconds=23.930),
    "pam_comm_settings": dict(matrix=1989, vars=10_759,  words=4_185_061,  seconds=29.910),
    "network":           dict(matrix=1801, vars=707_402, words=6_226_057,  seconds=42.380),
}
 
# Scenarios whose reference build time exceeds this are skipped under --fast.
SLOW_SECONDS = 60.0
 
 
@dataclass
class Scenario:
    """One benchmark configuration: how to build it, and how to label it."""
 
    key: str
    label: str            # as printed in the paper's "scenario" column
    level: str            # as printed in the paper's "level" column
    make: Callable[[], tuple[list, Algebra, dict]]
    notes: str = ""
 
 
# --------------------------------------------------------------------------
# The eight configurations.  Each ``make`` returns (monomials, algebra, kwargs)
# where kwargs are forwarded to MomentProblem.
# --------------------------------------------------------------------------
 
def npa_bipartite() -> tuple[list, Algebra, dict]:
    """Standard bipartite Bell scenario, 5 settings x 3 outcomes per side.
 
    15 projective operators per side, Alice commuting with Bob, built at
    NPA level 1 + AB.  State (NPA) moments, so cyclicity is off.
    """
    ops = OperatorSet()
    A = ops.add_povm_family(5, 3)
    B = ops.add_povm_family(5, 3)
    a = [m for row in A for m in row]          # 15 operators
    b = [m for row in B for m in row]          # 15 operators
    ops.declare_commuting(a, b)                # distinct Hilbert spaces
 
    mons = a + b
    mons += [[u, v] for u in a for v in b]     # + AB
    return mons, ops.algebra(), dict(dim=1, cyclicity=False)
 
 
def pam_plain() -> tuple[list, Algebra, dict]:
    """Prepare-and-measure, 14 preparations and one 14-outcome measurement.
 
    The preparations are idempotent (pure states) and declared mutually
    commuting; the guessing measurement is projective.  Third order in the
    preparations.  Tracial moments, so cyclicity stays at its default True.
    """
    ops = OperatorSet()
    R = ops.add_family(14, idempotent=True)
    M = ops.add_povm(14)
    ops.declare_commuting(R, R)
 
    mons = list(R) + list(M)
    mons += [[r, m] for r in R for m in M]
    mons += [[r, s] for r in R for s in R]
    mons += [[r, s, t] for r in R for s in R for t in R]   # third order
    return mons, ops.algebra(), dict(dim=1)
 
 
def pam_comm_states() -> tuple[list, Algebra, dict]:
    """As ``pam_plain``, but Bob holds a 2-setting, 2-outcome family.
 
    Same 14 commuting preparations; the single 14-outcome POVM is replaced by
    a genuine measurement family, which is what makes {rho_x} behave as an
    ensemble rather than an arbitrary operator set.
    """
    ops = OperatorSet()
    R = ops.add_family(14, idempotent=True)
    M = ops.add_povm_family(2, 2)
    m_flat = [m for row in M for m in row]                 # 4 operators
    ops.declare_commuting(R, R)
 
    mons = list(R) + m_flat
    mons += [[r, m] for r in R for m in m_flat]
    mons += [[r, s] for r in R for s in R]
    mons += [[r, s, t] for r in R for s in R for t in R]
    return mons, ops.algebra(), dict(dim=1)
 
 
def npa_tripartite() -> tuple[list, Algebra, dict]:
    """Tripartite Bell scenario, 4 settings x 4 outcomes per party.
 
    16 operators per party, every pair of distinct parties commuting, built
    from cross-party pairs only: 1 + AB + AC + BC.
    """
    ops = OperatorSet()
    parties = []
    for _ in range(3):
        fam = ops.add_povm_family(4, 4)
        parties.append([m for row in fam for m in row])    # 16 operators each
    a, b, c = parties
    ops.declare_commuting(a, b)
    ops.declare_commuting(a, c)
    ops.declare_commuting(b, c)
 
    mons = a + b + c
    for left, right in ((a, b), (a, c), (b, c)):
        mons += [[u, v] for u in left for v in right]
    return mons, ops.algebra(), dict(dim=1, cyclicity=False)
 
 
def npa_hybrid() -> tuple[list, Algebra, dict]:
    """Bipartite Bell scenario with heterogeneous outcome counts.
 
    Built from individual ``add_povm`` calls rather than one
    ``add_povm_family``: Alice has 5 settings with 2, 3, 4, 5 and 6 outcomes
    (20 operators), Bob has 7 settings with 2 through 8 outcomes (35
    operators).  Level 1 + AB.
    """
    ops = OperatorSet()
    a = [lab for n in (2, 3, 4, 5, 6) for lab in ops.add_povm(n)]        # 20
    b = [lab for n in (2, 3, 4, 5, 6, 7, 8) for lab in ops.add_povm(n)]  # 35
    ops.declare_commuting(a, b)
 
    mons = a + b
    mons += [[u, v] for u in a for v in b]
    return mons, ops.algebra(), dict(dim=1, cyclicity=False)
 
 
def pam_povm_prep() -> tuple[list, Algebra, dict]:
    """Prepare-and-measure with an untrusted preparation device.
 
    The plain family of pure states is replaced by a steering-type assemblage
    of 6 inputs and 6 outcomes -- 36 general POVM elements, declared neither
    idempotent nor orthogonal -- measured through one trusted 8-outcome
    guessing POVM.  Second order.
    """
    ops = OperatorSet()
    R = ops.add_povm_family(6, 6, idempotent=False, orthogonal=False)
    r_flat = [m for row in R for m in row]                 # 36 operators
    M = ops.add_povm(8)                                    # trusted, projective
 
    mons = r_flat + list(M)
    mons += [[r, s] for r in r_flat for s in r_flat]
    mons += [[r, m] for r in r_flat for m in M]
    return mons, ops.algebra(), dict(dim=1)
 
 
def pam_comm_settings() -> tuple[list, Algebra, dict]:
    """Prepare-and-measure with partially compatible measurement settings.
 
    12 preparations and a family of 4 settings x 2 outcomes.  Two pairs of
    Bob's settings are declared pairwise commuting (0--1 and 2--3); the
    remaining cross-pairs stay incompatible.  Third order.
 
    As in the other prepare-and-measure rows the preparations are declared
    mutually commuting.  That declaration is what collapses the variable count
    to five figures: it merges the permutations of each third-order word, all
    of which are already present in the monomial list, so the number of
    distinct words explored is unchanged while the number of classes drops by
    a factor of roughly twenty.
    """
    ops = OperatorSet()
    R = ops.add_family(12, idempotent=True)
    M = ops.add_povm_family(4, 2)
    m_flat = [m for row in M for m in row]                 # 8 operators
    ops.declare_commuting(R, R)
    ops.declare_commuting(M[0], M[1])                      # settings 0--1
    ops.declare_commuting(M[2], M[3])                      # settings 2--3
 
    mons = list(R) + m_flat
    mons += [[r, m] for r in R for m in m_flat]
    mons += [[r, s] for r in R for s in R]
    mons += [[r, s, t] for r in R for s in R for t in R]
    return mons, ops.algebra(), dict(dim=1)
 
 
def network() -> tuple[list, Algebra, dict]:
    """Small network scenario: three independent 24-outcome POVMs.
 
    A and B are two independently-prepared subsystems and are declared to
    commute with each other, but neither commutes with C, the central node's
    joint measurement -- the structure of the joint-measurement station in an
    entanglement-swapping protocol.  Second order, cross terms only.
    """
    ops = OperatorSet()
    A = ops.add_povm(24)
    B = ops.add_povm(24)
    C = ops.add_povm(24)
    ops.declare_commuting(A, B)                # independent sources
    # A--C and B--C are deliberately left incompatible.
 
    mons = list(A) + list(B) + list(C)
    for left, right in ((A, B), (A, C), (B, C)):
        mons += [[u, v] for u in left for v in right]
    return mons, ops.algebra(), dict(dim=1, cyclicity=False)
 
 
SCENARIOS: list[Scenario] = [
    Scenario("npa_bipartite",     "NPA (bipartite)",        "1+AB",      npa_bipartite),
    Scenario("pam_plain",         "PAM (plain)",            "3rd order", pam_plain),
    Scenario("pam_comm_states",   "PAM (commuting states)", "3rd order", pam_comm_states),
    Scenario("npa_tripartite",    "NPA (tripartite)",       "2nd order", npa_tripartite),
    Scenario("npa_hybrid",        "NPA (hybrid outcomes)",  "1+AB",      npa_hybrid),
    Scenario("pam_povm_prep",     "PAM (POVM preparation)", "2nd order", pam_povm_prep),
    Scenario("pam_comm_settings", "PAM (commuting settings)", "3rd order", pam_comm_settings),
    Scenario("network",           "Network (two channels)", "2nd order", network),
]
 
 
# --------------------------------------------------------------------------
# Running and reporting
# --------------------------------------------------------------------------
 
@dataclass
class Result:
    scenario: Scenario
    matrix: int
    n_vars: int
    words: int
    seconds: float
    all_times: list[float] = field(default_factory=list)
 
 
def run_scenario(sc: Scenario, repeat: int = 1, progress: bool = False) -> Result:
    """Build one scenario ``repeat`` times and keep the fastest build."""
    monomials, algebra, kwargs = sc.make()
    times: list[float] = []
    mm = None
    for _ in range(max(1, repeat)):
        problem = MomentProblem(monomials, algebra, **kwargs)
        t0 = time.perf_counter()
        mm = problem.build(progress=progress)
        times.append(time.perf_counter() - t0)
    stats = mm.stats
    return Result(
        scenario=sc,
        matrix=mm.n,
        n_vars=mm.n_variables,
        words=stats["distinct_words"],
        seconds=min(times),
        all_times=times,
    )
 
 
def environment_report() -> str:
    import numpy as np
    import MoMPy
 
    cpu = platform.processor() or platform.machine()
    # /proc/cpuinfo gives a far more useful model string on Linux.
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    lines = [
        "Environment",
        "-----------",
        f"  CPU      : {cpu}",
        f"  platform : {platform.platform()}",
        f"  python   : {platform.python_version()} ({platform.python_implementation()})",
        f"  numpy    : {np.__version__}",
        f"  MoMPy    : {getattr(MoMPy, '__version__', 'unknown')}",
    ]
    return "\n".join(lines)
 
 
HEADER = f"{'scenario':<26}{'level':<11}{'matrix':>12}{'vars':>10}{'words':>14}{'t (s)':>10}"
 
 
def format_row(r: Result) -> str:
    side = f"{r.matrix}x{r.matrix}"
    return (
        f"{r.scenario.label:<26}{r.scenario.level:<11}{side:>12}"
        f"{r.n_vars:>10,}{r.words:>14,}{r.seconds:>10.3f}"
    )
 
 
def format_latex_row(r: Result) -> str:
    def group(n: int) -> str:
        return f"{n:,}".replace(",", "\\,")
    return (
        f"{r.scenario.label} & {r.scenario.level} & "
        f"${r.matrix}\\times{r.matrix}$ & ${group(r.n_vars)}$ & "
        f"${group(r.words)}$ & ${r.seconds:.3f}$ \\\\"
    )
 
 
def check_against_paper(results: list[Result]) -> int:
    """Compare the deterministic columns against the published values."""
    print()
    print("Check against the values printed in the paper")
    print("--------------------------------------------")
    failures = 0
    for r in results:
        ref = PAPER.get(r.scenario.key)
        if ref is None:
            print(f"  {r.scenario.label:<26} no reference value")
            continue
        bad = []
        for name, got in (("matrix", r.matrix), ("vars", r.n_vars), ("words", r.words)):
            if got != ref[name]:
                bad.append(f"{name}: got {got:,}, paper {ref[name]:,}")
        if bad:
            failures += 1
            print(f"  MISMATCH {r.scenario.label:<20} " + "; ".join(bad))
        else:
            ratio = r.seconds / ref["seconds"] if ref["seconds"] else float("nan")
            print(
                f"  ok       {r.scenario.label:<20} "
                f"matrix/vars/words all match; "
                f"t = {r.seconds:.3f}s vs {ref['seconds']:.3f}s in the paper "
                f"({ratio:.2f}x)"
            )
    if failures:
        print(f"\n{failures} scenario(s) disagree on a deterministic column.")
    else:
        print("\nAll deterministic columns reproduce exactly.")
    return failures
 
 
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild the moment-matrix benchmarks of the MoMPy tutorial.",
    )
    ap.add_argument("-s", "--scenario", action="append", metavar="KEY",
                    help="run only this scenario (repeatable); see --list")
    ap.add_argument("--list", action="store_true", help="list scenario keys and exit")
    ap.add_argument("--fast", action="store_true",
                    help=f"skip scenarios taking more than ~{SLOW_SECONDS:.0f}s in the paper")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="build each scenario N times and report the fastest (default 1)")
    ap.add_argument("--latex", action="store_true",
                    help="also emit LaTeX rows for the tutorial's table")
    ap.add_argument("--check", action="store_true",
                    help="compare deterministic columns against the published values")
    ap.add_argument("--env", action="store_true",
                    help="print the environment report and exit")
    ap.add_argument("--progress", action="store_true",
                    help="show MoMPy's own per-build progress line")
    args = ap.parse_args(argv)
 
    if args.env:
        print(environment_report())
        return 0
 
    if args.list:
        for sc in SCENARIOS:
            ref = PAPER.get(sc.key, {})
            print(f"  {sc.key:<20} {sc.label:<26} "
                  f"(paper: {ref.get('seconds', float('nan')):.3f}s)")
        return 0
 
    selected = SCENARIOS
    if args.scenario:
        wanted = set(args.scenario)
        unknown = wanted - {sc.key for sc in SCENARIOS}
        if unknown:
            ap.error(f"unknown scenario key(s): {', '.join(sorted(unknown))}")
        selected = [sc for sc in SCENARIOS if sc.key in wanted]
    if args.fast:
        selected = [sc for sc in selected
                    if PAPER.get(sc.key, {}).get("seconds", 0.0) <= SLOW_SECONDS]
 
    print(environment_report())
    print()
    print("Timing MomentProblem.build() only; no SDP is solved.")
    print()
    print(HEADER)
    print("-" * len(HEADER))
 
    results: list[Result] = []
    total = 0.0
    for sc in selected:
        r = run_scenario(sc, repeat=args.repeat, progress=args.progress)
        results.append(r)
        total += r.seconds
        print(format_row(r), flush=True)
 
    print("-" * len(HEADER))
    print(f"{'total':<26}{'':<11}{'':>12}{'':>10}{'':>14}{total:>10.3f}")
 
    if args.latex:
        print()
        print("% --- rows for the tutorial's Table III ---")
        for r in results:
            print(format_latex_row(r))
 
    if args.check:
        return 1 if check_against_paper(results) else 0
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
