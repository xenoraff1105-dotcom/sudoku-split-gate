# Hypothesis 4 · Split gating in a looped Transformer for Sudoku

[中文](README.md) · English

**The experiments are complete. Further work on this design is on hold.** Can a recurrent model preserve correct answers and still fix mistakes as it runs for more rounds? The split-gated model (`split_gate`) gates writes to the first 32 channels of a 96-dimensional residual state and lets the other 64 update freely. The full-gating control (`full_gate`) computes 32 gate values per cell and applies each to three channels, covering all 96. Both models have 82,697 parameters.

On a fixed set of 512 hard puzzles, split gating changed fewer correct answers to incorrect ones in eight of ten training seeds. It also had a lower final correction rate in eight of ten seeds. This implementation fell short of the goal of protecting correct answers while preserving the ability to correct errors. The finding applies to the Sudoku task, channel split, and training protocol tested here. [Ten-seed report (Chinese)](sudoku_10seeds/sudoku_10seeds_report.md)

## Paired results on hard puzzles

Each model trained for 5,000 optimizer steps with 8 recurrent rounds per step. At test time, the model ran for 32 rounds. Cell-level comparisons match the two models on the same training seed, puzzle, and originally blank cell. Each seed has equal weight. Differences and their 95% bootstrap intervals are in percentage points, calculated as split gating minus full gating.

| Outcome | Split gating | Full gating | Difference | 95% interval, resampling seeds on fixed puzzles | 95% interval, resampling seeds and whole puzzles |
|---|---:|---:|---:|---:|---:|
| Both correct at round 8, then ever incorrect in rounds 9–32 | 5.10% | 5.51% | −0.40 | [−0.74, −0.09] | [−0.86, +0.04] |
| Both incorrect at round 8, correct at round 32 | 27.98% | 30.08% | −2.11 | [−3.47, −0.88] | [−4.32, +0.08] |
| Entire puzzle solved at round 32 | 41.64% | 43.32% | −1.68 | [−3.79, +0.16] | [−4.71, +1.17] |

The table uses 100,000 bootstrap draws. For the final correction-rate difference, the upper limit of the joint seed-and-puzzle interval was −0.005 percentage points with the prespecified 4,000 draws and +0.075 with 100,000 draws. The data therefore leave open a near-zero difference on new puzzles. Among the seven additional seeds, split gating changed correct answers to incorrect ones less often in five seeds (mean difference −0.32 percentage points), and its final correction rate was lower in six (mean difference −2.74 percentage points). All seven used the same test puzzles as the first three. [Per-seed metrics](sudoku_10seeds/sudoku_10seeds_metrics.csv) · [Bootstrap precision check](sudoku_10seeds/sudoku_10seeds_precision_check.json)

The round-32 counts show where the final difference came from. Among cells both models got right at round 8, split gating preserved 312 more correct answers at round 32. Among cells both got wrong, it corrected 749 fewer. Among cells where only one model was right at round 8, split gating finished with 215 fewer correct answers. Across 247,670 seed–cell observations, split gating finished with **652 fewer correct cells**. Across 5,120 seed–puzzle observations, it solved **86 fewer puzzles**. The original assessment's figure of 699 counts the difference in cells that became incorrect at *any* later round. The assessment also proposes shared normalization and read/write paths as possible mechanisms. These experiments did not directly measure those mechanisms. [Original assessment (Chinese)](sudoku_split_gate_opinion.md) · [Per-round predictions](sudoku_10seeds/sudoku_10seeds_predictions.npz)

## Experiments and artifacts

| Stage | Design | Files |
|---|---|---|
| Pilot | Verified the Sudoku training pipeline on Colab and measured runtime and compute units. | [Notebook](sudoku_gate_pilot.ipynb) · [Pilot results](sudoku_pilot_results.json) |
| Three-model comparison | Trained an ungated looped Transformer, a full-gated model, and a split-gated model with three seeds each. Compared rounds 8, 16, and 32 on the original test set. | [Notebook](sudoku_gate_formal.ipynb) · [Metrics](sudoku_formal_metrics.csv) · [Full results](sudoku_formal_results.json) |
| Hard-puzzle stress test | Fixed two sets of 512 new puzzles, each with a unique solution. Naked and hidden singles alone solved 506 puzzles from the original distribution and stalled on all 512 hard puzzles. Saved model weights and predictions for all 32 rounds for three models and three seeds. | [Notebook](sudoku_stress/sudoku_stress.ipynb) · [Summary](sudoku_stress/stress_summary.json) · [Stress-test files](sudoku_stress/) |
| Ten-seed replication | Reused the six gated models trained with seeds 17, 29, and 43. Trained both gated variants with seeds 59, 71, 83, 97, 109, 127, and 139 for 14 additional runs. | [Notebook](sudoku_10seeds/sudoku_10seeds.ipynb) · [Full statistics](sudoku_10seeds/sudoku_10seeds_summary.json) · [20 model checkpoints](sudoku_10seeds/sudoku_10seeds_weights.zip) |

## Reproduce the analysis offline

From the repository root, run the command below in a Python environment with NumPy installed. The script reads the puzzles, their unique solutions, and the per-cell predictions. It recomputes the paired metrics on CPU for both test sets and all ten seeds. The default is 4,000 bootstrap draws. Add `--bootstrap-samples 100000` to reproduce the intervals in the table.

```bash
python sudoku_10seeds/sudoku_10seeds_analysis.py sudoku_10seeds/sudoku_10seeds_predictions.npz --output recomputed.json
```

The published copies of the four notebooks omit account identifiers and time zones from Colab execution metadata. They retain the code and cell outputs, including per-round predictions and weights. Hashes in the experiment summaries refer to the original files. The manifest lists both the original and published SHA-256 digests. [Archive checksums](archive_sha256.json)
