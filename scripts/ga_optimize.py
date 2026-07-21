"""Genetic-algorithm parameter search with an anti-overfitting harness.

A GA is, by construction, an overfitting *maximiser*: its whole job is to
find the parameter vector that scores highest on the data it is shown. So
the GA on its own does not protect against overfitting -- the experimental
design around it does. This script wraps the GA in three defences:

1. THREE CHRONOLOGICAL WINDOWS, never shuffled:
     IS-train  ->  IS-validation  ->  OOS
   - The GA only ever *evolves* against IS-train fitness.
   - The winning genome is *selected* by IS-validation fitness (the genome
     with the best validation score across all generations -- an
     early-stopping / model-selection set, exactly like a validation set in
     ML). The best-on-train genome (the maximally-overfit one) is reported
     separately, only for contrast.
   - OOS (the last 5 years, held out project-wide) is touched EXACTLY ONCE,
     at the very end, as a pass/fail gate. Nothing is re-tuned after seeing
     it. If OOS ~= IS, the edge is plausibly real; if OOS << IS, the search
     overfit.

2. A DELIBERATELY TINY SEARCH SPACE (5 knobs). The high-leverage regime
   lever (pivot_period) and the rationale-free surfaces (MACD periods, cost
   assumptions, position size) are FROZEN, not searched -- see PARAM NOTES
   at the bottom of this file for the full change/don't-change rationale.
   Overfitting risk scales with the number and leverage of free parameters;
   the cheapest defence is to not hand the optimiser knobs it shouldn't turn.

3. A ROBUST FITNESS that can't be won cheaply: R-multiple profit factor
   (scale-invariant, no compounding distortion) aggregated across the whole
   pair universe, with a hard penalty for too few trades and a multiplier
   rewarding cross-pair consistency -- so a config can't score by overfitting
   a single lucky pair or a dozen outlier trades.

Run:  python scripts/ga_optimize.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.pivot_bounce_strategy import generate_signals
from utils import infer_pip_size

# --- what the GA is allowed to optimise (everything else is frozen) ----------
# Primary universe: the 5 majors the strategy was originally built and
# validated on -- where any real edge actually lives. (Swap for all 28 by
# editing this list; the crosses have already been shown not to generalise,
# so this is the fair "optimise the thing that works" test.)
UNIVERSE = ["EURUSD240", "GBPUSD240", "USDJPY240", "AUDUSD240", "USDCHF240"]

OOS_YEARS = 5          # held-out tail, project-wide convention
VAL_FRACTION = 0.35    # most-recent 35% of IS becomes the validation window
SEED = 20260721

# FROZEN (see PARAM NOTES): pivot period, MACD periods, spread/cost, risk,
# and every structural boolean flag in generate_signals.
PIVOT_PERIOD = "W"
SPREAD_PIPS = 2.0
RISK_PER_TRADE = 0.01
CAPITAL = 10_000

# GA hyperparameters
POP = 24
GENS = 12
ELITES = 2
TOURNAMENT_K = 3
MUT_PROB = 0.30

# genome bounds: (name, kind, lo, hi)
GENES = [
    ("tolerance_pips", "float", 3.0, 50.0),
    ("confirmation_window", "int", 1, 6),
    ("min_reward_risk", "float", 0.0, 2.0),
    ("stop_levels", "int", 1, 3),
    ("target_levels", "int", 1, 3),
]


# --- window-sliced data cache ------------------------------------------------
# Indicators are computed once on the FULL series and then sliced by window,
# so there is no EMA/pivot warm-up artefact at the internal train/val/OOS
# boundaries (which there would be if each window were loaded and warmed up
# independently).
_cache = {}  # pair -> dict(window -> (df, macd, pivots, pip))


def _build_cache():
    for pair in UNIVERSE:
        full = load_ohlcv_csv(f"data/raw/{pair}.csv")
        pip = infer_pip_size(full["close"])
        macd = compute_macd(full["close"])
        pivots = compute_daily_pivots(full, period=PIVOT_PERIOD)

        oos_cut = full.index[-1] - pd.DateOffset(years=OOS_YEARS)
        is_start = full.index[0]
        is_span = oos_cut - is_start
        val_cut = is_start + is_span * (1.0 - VAL_FRACTION)

        masks = {
            "train": (full.index >= is_start) & (full.index < val_cut),
            "val": (full.index >= val_cut) & (full.index < oos_cut),
            "oos": full.index >= oos_cut,
        }
        _cache[pair] = {}
        for w, mask in masks.items():
            _cache[pair][w] = (full[mask], macd[mask], pivots[mask], pip)
    # report the split once
    any_pair = UNIVERSE[0]
    for w in ("train", "val", "oos"):
        d = _cache[any_pair][w][0]
        print(f"  {w:<5} {d.index[0].date()} -> {d.index[-1].date()}  ({len(d)} bars, e.g. {any_pair})")


def _r_multiples(trades):
    out = []
    for t in trades:
        risked = t.size * abs(t.entry_price - t.stop)
        if risked > 0:
            out.append(t.pnl / risked)
    return out


def _r_pf(rs):
    if not rs:
        return float("nan")
    gp = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r <= 0)
    return gp / gl if gl > 0 else float("inf")


def evaluate(genome, window):
    """Run `genome` across the whole universe on one window; return the raw
    metrics plus the scalar fitness the GA maximises."""
    all_r, n_trades, pairs_pos = [], 0, 0
    dollar_pfs, rets = [], []
    for pair in UNIVERSE:
        df, macd, pivots, pip = _cache[pair][window]
        signals = generate_signals(
            df, macd, pivots,
            tolerance=genome["tolerance_pips"] * pip,
            confirmation_window=int(genome["confirmation_window"]),
            min_reward_risk=genome["min_reward_risk"],
            stop_levels=int(genome["stop_levels"]),
            target_levels=int(genome["target_levels"]),
        )
        bt = Backtester(
            initial_capital=CAPITAL, risk_per_trade=RISK_PER_TRADE,
            spread=SPREAD_PIPS * pip, commission=0.0, block_same_bar_reversal=True,
        )
        result = bt.run(df, signals)
        m = compute_metrics(result["trades"], result["equity_curve"], CAPITAL)
        pair_r = _r_multiples(result["trades"])
        all_r.extend(pair_r)
        n_trades += m["n_trades"]
        prpf = _r_pf(pair_r)
        pairs_pos += 1 if (np.isfinite(prpf) and prpf >= 1.0) else 0
        dollar_pfs.append(m["profit_factor"] if np.isfinite(m["profit_factor"]) else 0.0)
        rets.append(m["total_return_pct"])

    rpf = _r_pf(all_r)
    frac_pos = pairs_pos / len(UNIVERSE)
    min_trades = 5 * len(UNIVERSE)  # ~5 trades/pair minimum on the window
    if n_trades < min_trades or not np.isfinite(rpf):
        fitness = -1.0  # degenerate: too few trades to trust
    else:
        # reward cross-pair consistency so a single lucky pair can't win
        fitness = rpf * (0.5 + 0.5 * frac_pos)
    return dict(
        fitness=fitness, rpf=rpf, n_trades=n_trades,
        pairs_pos=pairs_pos, n_pairs=len(UNIVERSE),
        dollar_pf=float(np.mean(dollar_pfs)), avg_ret=float(np.mean(rets)),
    )


# --- GA plumbing -------------------------------------------------------------
def _key(g):
    return (round(g["tolerance_pips"], 2), int(g["confirmation_window"]),
            round(g["min_reward_risk"], 3), int(g["stop_levels"]), int(g["target_levels"]))


def _random_genome(rng):
    g = {}
    for name, kind, lo, hi in GENES:
        g[name] = rng.uniform(lo, hi) if kind == "float" else int(rng.integers(lo, hi + 1))
    return g


def _mutate(g, rng):
    out = dict(g)
    for name, kind, lo, hi in GENES:
        if rng.random() >= MUT_PROB:
            continue
        if kind == "float":
            step = (hi - lo) * 0.12
            out[name] = float(np.clip(out[name] + rng.normal(0, step), lo, hi))
        else:
            out[name] = int(np.clip(out[name] + rng.choice([-1, 1]), lo, hi))
    return out


def _crossover(a, b, rng):
    return {name: (a[name] if rng.random() < 0.5 else b[name]) for name, *_ in GENES}


def _tournament(pop, fits, rng):
    idxs = rng.choice(len(pop), size=TOURNAMENT_K, replace=False)
    best = max(idxs, key=lambda i: fits[i])
    return pop[best]


def _fmt(g):
    return (f"tol={g['tolerance_pips']:.1f} cw={int(g['confirmation_window'])} "
            f"mrr={g['min_reward_risk']:.2f} sl={int(g['stop_levels'])} tl={int(g['target_levels'])}")


def main():
    rng = np.random.default_rng(SEED)
    print(f"GA parameter search -- universe: {', '.join(UNIVERSE)}")
    print(f"Windows (chronological; OOS touched once at the end):")
    _build_cache()

    train_fit_cache = {}

    def train_fitness(g):
        k = _key(g)
        if k not in train_fit_cache:
            train_fit_cache[k] = evaluate(g, "train")
        return train_fit_cache[k]

    pop = [_random_genome(rng) for _ in range(POP)]
    best_val = None  # (val_metrics, genome, gen) -- selection by validation
    best_train = None  # (train_metrics, genome) -- overfit reference

    for gen in range(1, GENS + 1):
        fits_full = [train_fitness(g) for g in pop]
        fits = [f["fitness"] for f in fits_full]
        order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)

        # champion of this generation on TRAIN -> score it on VALIDATION
        champ = pop[order[0]]
        champ_train = fits_full[order[0]]
        champ_val = evaluate(champ, "val")

        if best_train is None or champ_train["fitness"] > best_train[0]["fitness"]:
            best_train = (champ_train, dict(champ))
        if best_val is None or champ_val["fitness"] > best_val[0]["fitness"]:
            best_val = (champ_val, dict(champ), gen)

        print(f"gen {gen:2d}: train rPF={champ_train['rpf']:.3f} fit={champ_train['fitness']:.3f} "
              f"| val rPF={champ_val['rpf']:.3f} fit={champ_val['fitness']:.3f} "
              f"pairs+={champ_val['pairs_pos']}/{champ_val['n_pairs']} | {_fmt(champ)}")

        # next generation: elitism + tournament/crossover/mutation
        nxt = [dict(pop[order[i]]) for i in range(ELITES)]
        while len(nxt) < POP:
            pa = _tournament(pop, fits, rng)
            pb = _tournament(pop, fits, rng)
            child = _mutate(_crossover(pa, pb, rng), rng)
            nxt.append(child)
        pop = nxt

    # --- final report --------------------------------------------------------
    val_m, val_g, val_gen = best_val
    train_m, train_g = best_train
    baseline = dict(tolerance_pips=18.0, confirmation_window=2, min_reward_risk=0.5,
                    stop_levels=2, target_levels=2)

    print("\n" + "=" * 92)
    print("SELECTED by validation (the early-stopped, overfitting-resistant pick):")
    print(f"  genome (found gen {val_gen}): {_fmt(val_g)}")
    print("Reference -- best-on-TRAIN (maximally overfit, shown only for contrast):")
    print(f"  genome: {_fmt(train_g)}")
    print("=" * 92)

    rows = [
        ("baseline (hand-tuned, round 3)", baseline),
        ("GA best-on-validation (SELECTED)", val_g),
        ("GA best-on-train (overfit ref)", train_g),
    ]
    print(f"\n{'config':<34}{'train rPF':>10}{'val rPF':>9}{'OOS rPF':>9}"
          f"{'OOS pairs+':>12}{'OOS trades':>11}")
    print("-" * 92)
    selected_oos = None
    for name, g in rows:
        tr = evaluate(g, "train")
        va = evaluate(g, "val")
        oo = evaluate(g, "oos")
        if name.startswith("GA best-on-validation"):
            selected_oos = oo
        print(f"{name:<34}{tr['rpf']:>10.3f}{va['rpf']:>9.3f}{oo['rpf']:>9.3f}"
              f"{oo['pairs_pos']:>9}/{oo['n_pairs']}{oo['n_trades']:>11}")

    base_oos = evaluate(baseline, "oos")
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    sel_train = evaluate(val_g, "train")
    decay = sel_train["rpf"] - selected_oos["rpf"]
    print(f"  Selected genome: train rPF {sel_train['rpf']:.3f} -> OOS rPF {selected_oos['rpf']:.3f} "
          f"(decay {decay:+.3f})")
    print(f"  Baseline OOS rPF: {base_oos['rpf']:.3f}  ({base_oos['pairs_pos']}/{base_oos['n_pairs']} pairs)")
    if selected_oos["rpf"] >= 1.0 and selected_oos["rpf"] >= base_oos["rpf"] - 0.01:
        print("  -> GA-tuned config holds up OOS AND is at least as good as the hand-tuned")
        print("     baseline: the search found real, generalising structure (within this universe).")
    elif selected_oos["rpf"] >= 1.0:
        print("  -> GA config is OOS-profitable but does NOT beat the hand-tuned baseline OOS:")
        print("     the extra search bought nothing that generalises -- baseline already captured it.")
    else:
        print("  -> GA config is NOT profitable OOS despite being tuned IS: the search OVERFIT.")
        print("     This is the expected failure mode and the reason the OOS gate exists.")
    print("\n  (rPF is the trustworthy metric. Dollar PF/return are compounding-distorted and")
    print("   were excluded from the objective on purpose.)")


if __name__ == "__main__":
    main()


# ============================================================================
# PARAM NOTES -- what the GA searches, what it must not, and why.
# ============================================================================
# SEARCHED (free genes) -- genuine low/moderate-leverage strategy knobs:
#   * tolerance_pips        3-50   noise band around a pivot for a "touch".
#                                  Real meaning, bounded, ablation-low-sensitivity.
#   * confirmation_window   1-6    bars a touch stays valid for a MACD cross.
#                                  Small integer range, real meaning.
#   * min_reward_risk       0-2    R:R entry filter. Near-neutral in ablation,
#                                  so safe to let float (low leverage).
#   * stop_levels           1-3    pivot steps to the stop -> risk geometry.
#   * target_levels         1-3    pivot steps to the target -> reward geometry.
#   Rationale: overfitting capacity grows with (#free params x their leverage
#   x #configs tried). The GA maximises the last term, so we shrink the first
#   two: a 5-knob, mostly-low-sensitivity surface is small enough that IF a
#   robust edge exists the search finds it, and if not, the OOS gate exposes
#   the fit rather than being fooled by a huge search.
#
# FROZEN (never handed to the GA) -- and why:
#   * pivot_period = W   HIGHEST-leverage lever in the whole strategy and
#                        already exhaustively validated (D and M clearly worse
#                        IS *and* OOS across 28 pairs). Searching it invites the
#                        GA to pick D/M on a lucky IS-train slice. High impact +
#                        strong prior = lock it, don't gamble it.
#   * MACD 12/26/9       Three extra continuous DoF with NO economic reason to
#                        prefer any non-standard value. This is the textbook
#                        curve-fitting surface -- deviating from the convention
#                        is how you manufacture an IS-only edge. Keep standard.
#   * spread / cost      NEVER optimise your cost assumptions -- lowering them
#                        "improves" results by lying to yourself. If anything,
#                        stress costs upward, never down.
#   * risk_per_trade     Money management, not edge. In R-multiple terms the
#                        edge is invariant to it; optimising dollar returns over
#                        it just drives it to the max and inflates ruin risk.
#                        Choose it from risk tolerance, not from a backtest.
#   * structural flags   require_touch / require_macd_confirmation /
#                        require_no_overshoot / use_signal_exit /
#                        block_same_bar_reversal DEFINE what the strategy *is*.
#                        Round-4 ablation already showed they're ~neutral;
#                        letting the GA flip them would drift into a different,
#                        untested strategy identity and muddy "did tuning the 5
#                        continuous knobs help." Hold structure constant.
#   * the OOS window     Not a parameter -- the exam. The GA never sees it;
#                        looked at exactly once, at the end.
# ============================================================================
