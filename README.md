# MPCAWIN

The canonical runnable project is `Downloads/MPCAWIN-main`. The other copy at
`Downloads/hoids/MPCAWIN-main` is an older working copy and is not used here.

Repairs and measured results: [Báo cáo sửa dự án](BAO_CAO_SUA_DU_AN.md).
The installed project `.venv` is ready to run. For the tested 60/20 variant:

```bash
.venv/bin/python run_imprtotal.py --data ../hoids/datasets-main/ImprTotal.csv \
  --config configs/legacy_60_20.json --output-dir results/run
```

On this subset, seed 20 detects batch ID 94 (the 95th batch). This is an
explicit parameter extension. The author-code 20/5 baseline still misses.

## Install

Python 3.9 matches the paper runtime; the pinned dependencies support 3.9–3.12.
Create a venv with a compatible interpreter, then install from this directory:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

Conda alternative: `conda env create -f environment.yml`, then
`conda activate mpcawin-paper`. On this machine the project `.venv` was created
using `/opt/anaconda3/envs/mpcawin/bin/python` (Python 3.9.25).
VS Code defaults to `.venv/bin/python`; if an interpreter was already selected,
use **Python: Select Interpreter** to select this project environment.

## Using a Git clone

The repository includes dataset preparation scripts under `dataset_tools/`;
the test suite uses temporary synthetic data and needs no external dataset.
Install the package as above before running scripts or tests. Large dataset
CSVs are supplied separately. Set `--data` to your dataset path:

```bash
python run_imprtotal.py --data /path/to/ImprTotal.csv \
  --config configs/legacy_60_20.json --output-dir results/run
```

Place the matching `ImprTotal.manifest.json` beside the CSV, supply `--manifest`,
or provide an explicit `--boundary-row`. Create a manifest from existing subsets
with `python dataset_tools/create_imprtotal.py --describe-existing` and the
explicit `--file-2017`, `--file-2018`, and `--output` paths shown below.
The `../hoids/datasets-main/` paths in local examples describe this machine's
existing data layout; replace them with your own paths after cloning.

## Describe the existing dataset

This verifies the two sources' order and creates a manifest without rewriting
any existing CSV. Original sample indices/seed cannot be recovered from old CSVs
and are explicitly marked unknown. Newly generated subsets record these fields.

```bash
python dataset_tools/create_imprtotal.py --describe-existing \
  --file-2017 ../hoids/datasets-main/Impr2017_stratified.csv \
  --file-2018 ../hoids/datasets-main/Impr2018_stratified.csv \
  --output ../hoids/datasets-main/ImprTotal.csv
```

## Run

```bash
python run_imprtotal.py --data ../hoids/datasets-main/ImprTotal.csv \
  --config configs/paper.json --output-dir results/baseline
python run_imprtotal.py --data ../hoids/datasets-main/ImprTotal.csv \
  --config configs/experiments.json --output-dir results/experiments
```

The installed `mpcawin-run` command accepts the same arguments from any directory.
Output folders contain `batches.csv` and `summary.json` with config, hashes,
versions, imported source paths, evidence before drift reset and boundary matches.
`batch_id` is zero-based; `batch_number1` is one-based. Index delay excludes the
current detecting batch; consumed batch/record delay includes it. Events outside
an explicit matching horizon are unmatched, rather than automatically true positives.
For a CSV without a manifest, `--boundary-row` must be supplied explicitly.

`configs/paper.json` preserves the author-code baseline: c=2, angle window=10,
legacy KSWIN 20/5/.009, reference scaling and rounded decisions. It may miss drift
on the reconstructed local subset. `configs/experiments.json` labels alternative
windows, River, alert_rate, scaling and precision separately. These alternatives
are not claims of reproducing the paper unchanged.

Batch errors (nonfinite values, changed schema, too few samples, zero variance)
raise before changing an established concept. A partial final batch is rejected
by default; configs can explicitly select `drop` or `process`. Eigenvalue angle
alone does not guarantee detection of mean shifts or isotropic scaling.

## Sensitivity

```bash
python test_c_values.py --data ../hoids/datasets-main/ImprTotal.csv \
  --config configs/paper.json --seeds 0 1 2 20 --output-dir results/sensitivity
```

The c sweep is independent of experimental KSWIN window choices. Preprocessing
and event matching are shared with the runner. New dataset generation uses
`--out-dir` and refuses to overwrite files; inspect each script's `--help`.

Run full-stream archived-output parity and 11-seed experimental validation:

```bash
python diagnostics/validate_repair.py --data ../hoids/datasets-main/ImprTotal.csv
```

## Provenance

`src/mpcawin_original.py` remains unchanged. Original runner/source/output files
are retained under `baselines/before_repair`. See
[the paper review](KE_HOACH_SUA_THEO_BAI_BAO.md) and [initial investigation](DRIFT_REVIEW.md).
Author commit: `e810ffc5e1d6d11cc66b74faa3fee6774cd15f52`.
