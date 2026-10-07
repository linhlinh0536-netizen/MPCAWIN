# ImprTotal drift investigation

Historical audit of the pre-repair version. Repairs have since been implemented;
see [the repair results](BAO_CAO_SUA_DU_AN.md). Source descriptions and line numbers
below refer to the archived files under `baselines/before_repair/`.

Reviewed and reproduced on 2026-10-07. Production Python files and the existing
`mpcawin_results.csv` / `c_sensitivity_results.csv` were not modified.

Subsequent comparison with the supplied paper confirms that window=20/stat=5
and within-year shuffling match the published experiment. The missed drift
described below is a failure on this local subset, not evidence those choices
depart from the paper. The local KSWIN update logic also matches upstream
scikit-multiflow 0.5.3 on the checked stream. See
[the paper comparison and file repair plan](KE_HOACH_SUA_THEO_BAI_BAO.md) for
the updated interpretation; 60/20 and alert_rate are experimental variants,
not configurations that reproduce the paper unchanged.

## Which project is actually running?

The runner is `/Users/linhkhanh/Downloads/MPCAWIN-main/run_imprtotal.py`.
It prepends its own `src` directory to `sys.path` (line 14). It therefore uses
the SciPy-based local `src/kswin_compat.py`, NOT the River wrapper in
`/Users/linhkhanh/Downloads/hoids/MPCAWIN-main` that was standardized earlier.
`src/mpcawin_original.py` and `src/mpcawin.py` differ only in their KSWIN import.

Reviewed the entire runner, both MPCAWIN files, the local KSWIN implementation,
`test_c_values.py`, both saved result CSVs, and the dataset creation/batching
scripts relevant to this stream.

## Reproduction and evidence

Ran an unchanged copy of the original runner and local sources in a temporary
directory to preserve the user's original outputs. Exit code: 0.
Evidence is in `diagnostics/results/original_run.log` and `original_run.csv`.

- Dataset: `Downloads/hoids/datasets-main/ImprTotal.csv`.
- Shape: 320,000 rows, 93 columns; selected features: 84 numeric columns.
- Feature data: no NaN or infinity.
- 160 batches of 2,000 rows; no remainder.
- Batch indices are zero-based; 2018 starts at batch 80 / row 160,000.
- Runner parameters: `coef_std=3`, KSWIN `alpha=0.009`, `window_size=20`,
  `stat_size=5`.
- After warmup, batches 11-79 have 18 alerts among 69 evaluated batches.
- Batches 80-159 have **80 alerts among 80 evaluated batches**.
- Median angle: 8.07 degrees before the boundary (after warmup), 29.175 after.
- At batch 80, angle=29.95, threshold=11.98, alert=1, drift=False.
- There are zero final drift detections. Minimum observed KS p-value across the
  original run is 0.07936507936507936, which exceeds 0.009.

The PCA/threshold stage recognizes changed dissimilarity in this dataset.
The KSWIN confirmation stage fails to turn that signal into a drift event.

## Main cause: inadequate KS sample size for this binary signal

`src/mpcawin.py:20` configures 5 points per sample. The MPCAWIN code at line 192
feeds KSWIN a binary alert, not the continuous PCA angle.

Computed with the actual SciPy 1.13.1 `ks_2samp(method='exact')`:

| Two binary samples, each length 5 | KS statistic | Returned p-value |
| --- | ---: | ---: |
| Reference all 0; recent has four 1s and one 0 | 0.8 | 0.0793650794 |
| Reference all 0; recent all 1 | 1.0 | 0.00793650794 |

With alpha=0.009, even D=0.8 is rejected as insufficient evidence. For samples
of five binary values, the two alert rates must differ by 1 to pass the current
gate: one sample all 0 and the other all 1. This event never occurs in the
original seeded run. Raising alpha only to 0.05 would still not accept D=0.8.

Pre-boundary alerts contaminate the randomly sampled reference. For example,
at batch 84 the reference has two 1s and the recent sample has five 1s:
D=0.6, p=0.357143, drift=False.

KSWIN keeps sliding even without a confirmed drift. By batch 100 both samples
are `[1,1,1,1,1]`, so D=0 and p=1. Persistent alerts cannot rescue a missed
transition once both comparison windows contain the new distribution. The
SAD threshold remains 11.98 because alerts do not update its angle window;
this is the current intended behavior, not the reason KS returns p=1.

SciPy documents its KS p-value test for independent samples from continuous
distributions. Binary alerts have ties and require separate evaluation of
sensitivity and false detections; the nominal alpha should not be interpreted
as an exact Bernoulli false-positive guarantee.
Source: https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.ks_2samp.html

## Controlled full-stream comparisons

`diagnostics/check_imprtotal.py` reads the same dataset, selects the same 84
features, runs the actual local MPCAWIN code, and gives each configuration a
fresh detector. All runs keep min/max angle window=10, coef_std=3, seed=20.
These are complete stream runs including MPCAWIN resets after detections,
not merely replays of saved alerts.

| Confirmation stage | Detection batches | Events before batch 80 | Delay |
| --- | --- | --- | --- |
| Current local KSWIN: window=20, stat=5, alpha=.009 | None | None | Missed |
| Local KSWIN: window=60, stat=20, alpha=.009 | 94 | None | 14 batches / 28,000 records |
| River KSWIN: window=60, stat=20, alpha=.009, seed=20 | 95 | None | 15 batches / 30,000 records |
| Current alert_rate: window=20, alpha=.5 | 82 | None | 2 batches / 4,000 records |

Merely replaying the original alert stream through River with the original
window=20/stat=5 also produced no detections (seed=20). Changing backend alone
does not resolve this parameter problem.

The delay in records follows the runner's convention: batch-index delay times
batch size. It excludes the 2,000 records consumed to finish the detecting batch.
These results describe this dataset and seed; they are not general sensitivity
or false-positive estimates. Parameters must be validated on separate streams
and multiple seeds before adoption.

Run the comparison again from this project root:

```bash
/Users/linhkhanh/Downloads/hoids/MPCAWIN-main/.venv/bin/python \
  diagnostics/check_imprtotal.py \
  --data /Users/linhkhanh/Downloads/hoids/datasets-main/ImprTotal.csv
```

Full result tables and source paths are in `diagnostics/results/summary.json`
and the four configuration CSVs beside it.

## Other findings in the files

1. `src/kswin_compat.py:53-105` tests before appending the current alert. Its
   decision uses the previous alerts and starts testing one update later than
   an append-then-test implementation. This contributes latency but does not
   explain missing all 80 post-boundary alerts.
2. `src/kswin_compat.py:71` uses `np.random.choice` with replacement by default
   and the global NumPy RNG. Repeated reference values and the seed influence
   whether complete separation is ever sampled. This is a property of this
   legacy implementation, not proof it differs from scikit-multiflow.
3. `src/mpcawin.py:20` creates a shared default detector. Separate simultaneous
   instances can interfere. It is not the immediate cause in the single-model
   runner; sequential `test_c_values.py` runs reset the shared detector when
   setting their first concept.
4. `src/mpcawin.py:112` resets the alert window to max_window_size*2 instead of
   alert_window_size; lines 183-190 only check drift when already full before
   insertion. The default window=20 matches by coincidence in this experiment.
5. `src/mpcawin.py:174-178` needs vector-size checks, zero-norm handling and
   clipping before acos. No such numerical failure occurred on this run.
6. `run_imprtotal.py:289-319` reads KS debug data after MPCAWIN may reset it.
   On a detecting batch, p-value/statistic/reference can be erased before CSV
   recording. This does not prevent detection but hides its evidence. The
   comparison script preserves the decision p-value before reset.
7. `test_c_values.py` varies the PCA threshold coefficient, not the failed KS
   sampling parameters. Its stored c=2..5 runs still produce 20/20 alerts just
   after the boundary without detection. c=6 suppresses post-boundary alerts.
   Increasing coef_std therefore does not solve the observed confirmation issue.

## Dataset interpretation

`prepare_impr_stratified.py:325` shuffles each sampled year's data. The generator
keeps 160,000 rows per year with 3% attacks. `create_imprtotal.py:27` concatenates
2017 then 2018 without another shuffle. Both halves of the actual ImprTotal
file have attack ratio 0.03.

The boundary preserves a change of source distribution, but the shuffled files
do not preserve original within-year attack timelines. `split_batches.py` and
`run_imprtotal.py` hardcode the source boundary; they do not supply independently
annotated statistical drift ground truth. The observed feature/angle shift and
alerts support a detectable source change in this particular experiment.
Attack presence itself is not a drift annotation.

## Recommended repair order

1. Work on the actual runner project and make its imported source path and
   interpreter explicit. Keep the original results as the baseline.
2. Correct detector ownership, reset behavior, numerical guards and debug
   preservation independently of parameter tuning.
3. Validate KSWIN with a larger reference/statistical window; window=60/stat=20
   is a demonstrated candidate here. Compare it with the existing alert_rate
   strategy on held-out streams, stationary/noisy controls and multiple seeds.
4. Make runner parameters configurable, derive the source boundary from actual
   dataset construction, and label it as a source boundary rather than assuming
   every post-boundary detection is a validated true positive.
5. Report misses, events before the source change, detection delays and alert
   rates separately. Keep labels and the known boundary out of detector inputs.

No production algorithm or runner configuration was changed during this review.
