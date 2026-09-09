# Notebook Walkthrough — `template/notebook-v3.ipynb`

This document walks through every major section of the template notebook, explaining **what the code does**, **why it is there**, and **what you are expected to edit**.

---

## Overall design of the notebook

The notebook is a **guided executable experiment**, not a bag of independent snippets.  
Markdown cells teach the “why”; code cells do the work.  
Most of the time you only change a few configuration variables and then run top-to-bottom.

Custom code should stay rare. If Ultralytics already provides the functionality, use it.

---

## Section 0 — Environment

**Purpose**  
Make the runtime usable on Colab, Kaggle, or a local machine after a fresh start.

**What happens**

1. Upgrade Ultralytics (`%pip install -q -U ultralytics`) so you are not stuck on an old Colab image.
2. (Optional) clone the repository if you are starting from a blank Colab.
3. Import the few libraries the notebook needs.
4. Run `yolo checks` — always read the output (GPU visibility, disk space, versions).
5. Optionally set `ULTRALYTICS_API_KEY` and `datasets_dir`.

**What you edit**

- Uncomment and fill the API key if you want Platform streaming.
- Point `datasets_dir` at the real root of your data when paths in the YAML are relative.

**Why it matters**  
Training that silently falls back to CPU, or a dataset path that resolves to the wrong place, are two of the most common ways to waste hours.

---

## Section 1 — Experiment Setup

**Purpose**  
Declare, in one place, everything that defines “this experiment”.

**Key variables you must set**

```python
TASK = "detect"                 # detect | segment | semantic | depth | classify | pose | obb
MODEL = "yolo26n.pt"            # always a pretrained checkpoint
DATA = "configs/datasets/...."  # local yaml, classify folder, or ul:// URI
EXPERIMENT_NAME = "baseline"    # self-describing name
PROJECT = f"runs/{TASK}"        # or a Platform project slug
```

The notebook then loads any existing file:

```
configs/experiments/{EXPERIMENT_NAME}.yaml
```

and stores the result in `overrides`. Only intentional changes belong in that file.

**Sanity-check cell**  
It compares the model filename suffix (`-seg`, `-pose`, …) with the chosen `TASK`.  
A mismatch does **not** crash Ultralytics; it just trains against the wrong target and produces near-zero metrics. The warning catches the problem in seconds instead of after a full training run.

**What you edit**  
The five configuration variables above and, when needed, a small overrides YAML.

---

## Section 2 — Dataset Preparation & Validation

**Purpose**  
Catch dataset problems **before** you spend GPU hours.

The checks are ordered from cheapest to most conclusive:

1. **Structural validation** (`check_det_dataset`)  
   Does the YAML parse? Do the paths exist? (Milliseconds.)

2. **Visual spot-check** (detect only)  
   Draw one image + its labels using Ultralytics’ own path logic (`img2label_paths`).  
   This is the fastest human way to notice boxes that are offset or completely wrong.

3. **Smoke test** (`RUN_SMOKE_TEST = True`)  
   Train for 1 epoch on 10 % of the data.  
   Anything that would break a real run (missing files, wrong label format, kpt_shape mismatch, …) breaks here in under a minute.

4. **Inspect the free plots** that the smoke test already produced  
   - `labels.jpg` — class balance and box statistics  
   - `labels_correlogram.jpg` — systematic labeling artifacts  
   - `train_batch0.jpg` — actual augmented images with labels drawn on them

**What you edit**

- Set `RUN_SMOKE_TEST = True` the first time you use a new dataset (or after any label change).  
- Leave it `False` on subsequent runs once you trust the data.

**Philosophy**  
Reuse Ultralytics’ own machinery instead of writing a second, possibly buggy, dataset inspector.

---

## Section 3 — Model & Training

**Purpose**  
Load a pretrained checkpoint and run the actual training.

```python
model = YOLO(MODEL)          # always pretrained
model.train(
    data=DATA,
    project=PROJECT,
    name=EXPERIMENT_NAME,
    **overrides,
)
```

**Important design choices**

- The notebook never exposes `pretrained=False`. Starting from random weights is almost always the wrong choice for fine-tuning.
- Class-count mismatches are handled automatically by Ultralytics (the head is re-initialized; the backbone keeps its weights).
- Every run gets its own directory under `runs/` (`baseline`, `baseline2`, …). Nothing is overwritten.

**Optional tuning (baseline-first)**  
`RUN_TUNE = False` by default. The genetic tuner (`model.tune()`) runs many short trainings
and is the **least** effective lever. Exhaust cheaper improvements first (better labels,
longer training, larger image size, bigger model, domain augmentation) before enabling it.

Workflow enforced by the notebook:

1. Always complete the normal single-shot `model.train(...)` baseline first.
2. Inspect Section 4. Only if the baseline is healthy, set `RUN_TUNE = True` and run the
   tuning cell (`TUNE_EPOCHS`, `TUNE_ITERATIONS`, `TUNE_NAME`).
3. Results → `runs/<task>/<name>_tune/best_hyperparameters.yaml`.
4. Promote useful values into a new experiment YAML, choose a new `EXPERIMENT_NAME`,
   leave `RUN_TUNE = False`, and retrain fully. Do not ship the short-search checkpoint.

See the notebook intro section "Optional hyperparameter tuning (baseline-first workflow)",
the `yolo-tuning` skill, and the [Ultralytics guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

**What you edit**  
Usually nothing beyond the overrides file and the experiment name.  
Set `RUN_TUNE = True` only when you have already fixed the more important bottlenecks.

---

## Section 4 — Evaluation & Inspection

**Purpose**  
Evaluate the **best** checkpoint and understand the results beyond a single scalar.

```python
BEST = RUN_DIR / "weights" / "best.pt"
best_model = YOLO(BEST)
metrics = best_model.val(data=DATA, plots=True, save_json=True)
```

**Always use `best.pt`**, never `last.pt`.  
`last.pt` is only useful for resuming an interrupted run; it may already be over-fitted.

**Plots to read carefully**

- `results.png` — train vs. val curves (over- / under-fitting)
- `confusion_matrix.png` — which classes are confused
- `PR_curve.png` / `F1_curve.png` — how to choose a deployment confidence threshold
- A few raw prediction images — the cheapest way to discover failure modes that mAP hides

**What you edit**  
Rarely anything. You mainly inspect.

---

## Section 5 — Experiment Summary

**Purpose**  
Assemble configuration + metrics into one readable, re-runnable record.

The cell deliberately reads from disk (`args.yaml`, `results.csv`, `best.pt`) so it still works if you re-open the notebook in a fresh kernel later.

Everything printed here already exists on disk; the cell just makes it human-friendly.
If optional tuning was run, record the path to `best_hyperparameters.yaml` in the free-form
Notes of the Section 7 report.

---

## Section 6 — Export

**Purpose**  
Convert the trained model into a deployment format and verify that the conversion did not silently degrade accuracy.

```python
EXPORT_FORMAT = "onnx"   # or engine, coreml, openvino, …
export_path = best_model.export(format=EXPORT_FORMAT)
```

Then re-validate the exported artifact with the same `val()` call.

**What you edit**  
The string `EXPORT_FORMAT` and, if needed, extra export arguments (see the `yolo-export` skill).

---

## Section 7 — Report

**Purpose**  
Write a durable markdown file that survives kernel restarts and is readable by a teammate (or future you).

The report is written to:

```
docs/{EXPERIMENT_NAME}_report.md
```

It contains configuration, training summary, an optional-tuning subsection (whether
`RUN_TUNE` was enabled and where `best_hyperparameters.yaml` lives), metrics, export info,
and a free-form “Notes” section for qualitative observations.

---

## Mental model of data flow

```
configs/datasets/*.yaml          ─┐
configs/experiments/*.yaml       ─┤
                                 ├─► notebook (Section 1) ─► overrides dict
TASK / MODEL / DATA / NAME       ─┘
                                         │
                                         ▼
                              model.train(...)  ──► runs/<task>/<name>/
                                         │              ├── weights/best.pt
                                         │              ├── args.yaml
                                         │              ├── results.csv
                                         │              └── *.png plots
                                         ▼
                              val / predict / export
                                         │
                                         ▼
                              docs/<name>_report.md
```

---

## When you are allowed to write custom code

- Inside the notebook only in places the markdown itself marks as customization points.
- Under `scripts/data/` for one-off conversion (prefer Ultralytics converters first).
- Under `scripts/inference/` for deployment-specific helpers that Ultralytics does not cover.

Everywhere else, prefer the standard Ultralytics API.

---

## Quick reference — what to change for a new project

| Location | Typical change |
|----------|----------------|
| Section 1 variables | `TASK`, `MODEL`, `DATA`, `EXPERIMENT_NAME` |
| `configs/datasets/` | New dataset YAML |
| `configs/experiments/` | Optional overrides YAML |
| Section 2 | Set `RUN_SMOKE_TEST = True` once |
| Section 3 | Leave `RUN_TUNE = False` until baseline is healthy; then promote hyps |
| Section 6 | Change `EXPORT_FORMAT` if needed |
| Everything else | Leave alone |

That is the entire editing surface of the template.
