"""Paired loss, correction and whole-puzzle comparisons for the gate extension."""

import numpy as np


COHORT_NAMES = ("hard", "original")
GATE_VARIANTS = ("split_gate", "full_gate")
ENDPOINTS = ("ever_wrong", "correct32", "ever_correct", "board_acc32")
RESAMPLING = ("puzzles", "seeds", "crossed")


def load_previous_predictions(npz, cohorts, old_seeds=(17, 29, 43)):
    """Read both gate arms and verify the earlier NPZ uses the same puzzle/cell order."""
    matrices = {}
    with np.load(npz, allow_pickle=False) as archive:
        if not np.array_equal(archive["rounds"], np.arange(1, 33)):
            raise ValueError("Expected the original predicted digits for rounds 1 through 32")
        for cohort in COHORT_NAMES:
            x, y = cohorts[cohort]
            board, cell = np.nonzero(x == 0)
            fields = {"x": x, "y": y, "blank_board": board, "blank_cell": cell}
            for field, expected in fields.items():
                if not np.array_equal(archive[f"{cohort}_{field}"], expected):
                    raise ValueError(f"Prior {cohort}/{field} differs from this evaluation cohort")
            for model in GATE_VARIANTS:
                for seed in old_seeds:
                    matrices[cohort, model, seed] = archive[f"{cohort}_{model}_seed{seed}"]
    return matrices


def _puzzle_counts(x, y, matrices, cohort, seeds):
    if (x.ndim != 2 or x.shape != y.shape or x.shape[1] != 81 or not len(x)
            or x.dtype.kind not in "iu" or y.dtype.kind not in "iu"
            or x.min() < 0 or x.max() > 9 or y.min() < 1 or y.max() > 9
            or np.any((x != 0) & (x != y))):
        raise ValueError(f"Invalid {cohort} Sudoku boards or solutions")
    sizes = (x == 0).sum(axis=1)
    if np.any(sizes == 0):
        raise ValueError("Every puzzle must have at least one blank")
    truth = y[x == 0]
    offsets = np.cumsum(sizes) - sizes

    def per_puzzle(mask):
        return np.add.reduceat(mask, offsets, dtype=np.int64)

    counts = np.empty((len(seeds), len(x), len(ENDPOINTS), 3), dtype=np.int64)
    for i, seed in enumerate(seeds):
        flags = []
        for variant in GATE_VARIANTS:
            pred = np.asarray(matrices[cohort, variant, seed])
            if (pred.shape != (32, len(truth)) or pred.dtype.kind not in "iu"
                    or pred.min() < 1 or pred.max() > 9):
                raise ValueError(f"Invalid digit predictions for {cohort}/{variant}/{seed}")
            after = pred[8:] == truth
            flags.append((pred[7] == truth, after[-1], ~after.all(0), after.any(0)))
        s8, s32, s_lost, s_fixed = flags[0]
        f8, f32, f_lost, f_fixed = flags[1]
        right, wrong = s8 & f8, ~s8 & ~f8
        for j, (common, split, full) in enumerate(((right, s_lost, f_lost), (wrong, s32, f32),
                                                   (wrong, s_fixed, f_fixed))):
            counts[i, :, j] = np.column_stack((per_puzzle(common), per_puzzle(common & split),
                                               per_puzzle(common & full)))
        counts[i, :, 3] = np.column_stack((np.ones_like(sizes), per_puzzle(s32) == sizes,
                                          per_puzzle(f32) == sizes))
    return counts


def _rate_gaps(totals):
    den, split, full = np.moveaxis(totals, -1, 0)
    result = np.full(den.shape, np.nan)
    return np.divide(split - full, den, out=result, where=den != 0)


def _intervals(counts, resamples, seed):
    n_seeds, n_puzzles, n_endpoints = counts.shape[:3]
    flat = counts.transpose(1, 0, 2, 3).reshape(n_puzzles, -1)
    observed_gaps = _rate_gaps(counts.sum(axis=1))
    # Retain the original puzzle-bootstrap random stream so the original interval
    # can be reproduced exactly. Seed resampling uses an independent stream.
    puzzle_rng = np.random.default_rng(seed)
    seed_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    puzzle_prob = np.full(n_puzzles, 1 / n_puzzles)
    draws = {name: np.empty((resamples, n_endpoints)) for name in RESAMPLING}
    for start in range(0, resamples, 128):
        stop = min(resamples, start + 128)
        size = stop - start
        multiplicities = puzzle_rng.multinomial(n_puzzles, puzzle_prob, size=size)
        # A single bootstrap puzzle sample applies to every endpoint, arm, and
        # seed. Recompute each seed's conditional denominator on these puzzles.
        totals = (multiplicities @ flat).reshape(size, n_seeds, n_endpoints, 3)
        puzzle_gaps = _rate_gaps(totals)
        seed_indices = seed_rng.integers(n_seeds, size=(size, n_seeds))
        draws["puzzles"][start:stop] = puzzle_gaps.mean(1)
        draws["seeds"][start:stop] = observed_gaps[seed_indices].mean(1)
        draws["crossed"][start:stop] = puzzle_gaps[np.arange(size)[:, None], seed_indices].mean(1)
    result = []
    for j in range(n_endpoints):
        intervals, valid = {}, {}
        for name, values in draws.items():
            column = values[:, j]
            valid[name] = int(np.isfinite(column).sum())
            intervals[name] = (np.quantile(column, [0.025, 0.975]).tolist()
                               if valid[name] == resamples else None)
        result.append((intervals, valid))
    return result


def _summarize(counts, seeds, resamples, random_state):
    totals = counts.sum(axis=1)
    intervals = _intervals(counts, resamples, random_state)
    metrics = {}
    for j, endpoint in enumerate(ENDPOINTS):
        rows = []
        for seed, (den, split, full) in zip(seeds, totals[:, j].tolist()):
            rows.append(dict(seed=seed, denominator=den, split_events=split, control_events=full,
                             split_rate=split / den if den else None,
                             control_rate=full / den if den else None,
                             gap=(split-full) / den if den else None))
        means = {}
        for name in ("split_rate", "control_rate", "gap"):
            values = [r[name] for r in rows]
            means[name] = None if None in values else float(np.mean(values))
        combined = {name: sum(r[name] for r in rows)
                    for name in ("denominator", "split_events", "control_events")}
        gaps = np.array([r["gap"] for r in rows if r["gap"] is not None])
        limits = [float(gaps.min()), float(gaps.max())] if len(gaps) else None
        metrics[endpoint] = dict(by_seed=rows, mean_over_seeds=means, counts_across_seeds=combined,
                                 sign_counts=dict(negative=int((gaps < 0).sum()),
                                                  zero=int((gaps == 0).sum()),
                                                  positive=int((gaps > 0).sum()),
                                                  undefined=len(seeds)-len(gaps)),
                                 gap_range=limits, ci95=intervals[j][0], valid_draws=intervals[j][1])
    return dict(seeds=list(seeds), metrics=metrics)


def analyze_extension(cohorts, matrices, new_seeds, *, old_seeds=(17, 29, 43),
                      bootstrap_samples=4000, bootstrap_seed=2026):
    """Analyze split minus full gate on held-out original/hard puzzle trajectories."""
    old, new = tuple(map(int, old_seeds)), tuple(map(int, new_seeds))
    seeds = old + new
    if not old or not new or len(seeds) != len(set(seeds)):
        raise ValueError("Expected distinct nonempty groups of original and extension seeds")
    if not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    analyses = {}
    for cohort in COHORT_NAMES:
        x, y = map(np.asarray, cohorts[cohort])
        counts = _puzzle_counts(x, y, matrices, cohort, seeds)
        answers = dict(n_boards=len(x), n_blank=int(np.count_nonzero(x == 0)))
        for name, chosen, group in (("all", counts, seeds), ("new", counts[len(old):], new),
                                    ("previous", counts[:len(old)], old)):
            answers[name] = _summarize(chosen, group, bootstrap_samples, bootstrap_seed)
        analyses[cohort] = answers
    return dict(
        cohorts=analyses, models=list(GATE_VARIANTS), old_seeds=list(old), new_seeds=list(new),
        reference_round=8, follow_up_rounds=[9, 32], rate_scale="fraction", gap="split_gate minus full_gate",
        endpoints={
            "ever_wrong": "Among jointly correct R8 blanks, ever incorrect in rounds 9–32.",
            "correct32": "Among jointly incorrect R8 blanks, correct at round 32.",
            "ever_correct": "Among jointly incorrect R8 blanks, ever correct during rounds 9–32.",
            "board_acc32": "Of all puzzles, every originally blank cell correct at round 32."},
        bootstrap_samples=int(bootstrap_samples), bootstrap_seed=int(bootstrap_seed),
        interval_method="Marginal 95% percentile intervals for equal-seed mean differences. "
                        "Recompute common-blank denominators after each puzzle draw.",
        uncertainty={
            "puzzles": "Puzzle resampling conditional on the observed seeds.",
            "seeds": "Paired seed resampling conditional on these fixed test puzzles.",
            "crossed": "Independent resampling of paired seeds and whole puzzles; "
                       "the same puzzle IDs apply to every selected seed and model."},
        interpretation="The new seeds give a replication on the same previously examined test sets. "
                       "Seed inference relies on these replicates representing retraining variability. "
                       "Puzzle inference is limited to the respective generator and screening protocol. "
                       "Intervals are marginal by endpoint with no multiplicity adjustment. "
                       "An interval crossing zero does not establish equal correction rates.",
        noninferiority_margin=None,
    )


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="Combined gate predictions in the original trajectory layout")
    parser.add_argument("--old-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--new-seeds", type=int, nargs="+", help="Training seed order for this extension")
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="JSON destination (default: stdout)")
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as packed:
        cohorts = {name: (packed[f"{name}_x"], packed[f"{name}_y"]) for name in COHORT_NAMES}
        prefix = "hard_split_gate_seed"
        present = {int(k[len(prefix):]) for k in packed if k.startswith(prefix)}
    new = args.new_seeds if args.new_seeds is not None else sorted(present.difference(args.old_seeds))
    seeds = args.old_seeds + new
    if present != set(seeds):
        raise ValueError("Specified seeds differ from packed split-gate predictions")
    matrices = load_previous_predictions(args.npz, cohorts, seeds)
    analysis = analyze_extension(cohorts, matrices, new, old_seeds=args.old_seeds,
                                 bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    text = json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
