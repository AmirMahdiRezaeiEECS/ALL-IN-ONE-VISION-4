# How to Edit the Template for a New Project

This is the practical, step-by-step recipe.  
Follow it once and you will understand the pattern forever.

---

## Goal

Turn the generic `template/` into a concrete project such as:

```
projects/Detect/MyDataset/
```

or

```
projects/Segment/Cityscapes/
```

---

## Step 1 — Choose the task + dataset name

Decide two things:

1. **Task** — one of the values the notebook understands:
   - `detect`
   - `segment`
   - `semantic`
   - `depth`
   - `classify`
   - `pose`
   - `obb`

2. **Dataset** — a short, descriptive name (e.g. `VOC`, `MyCars`, `WarehouseBoxes`).

The resulting path will be:

```
projects/<Task>/<Dataset>/
```

Example: detection on Pascal VOC → `projects/Detect/VOC/`.

---

## Step 2 — Copy the entire template

From the repository root:

```bash
cp -r template projects/Detect/MyDataset
```

(You can also copy only the files you need, but copying the whole tree is safer and keeps the structure identical.)

After the copy you should see:

```
projects/Detect/MyDataset/
├── configs/
│   ├── datasets/
│   └── experiments/
├── datasets/
├── docs/
├── notebook-v3.ipynb      ← rename to notebook.ipynb if you prefer
├── runs/
├── scripts/
└── README.md
```

---

## Step 3 — Create (or adapt) the dataset YAML

Put a file under:

```
projects/Detect/MyDataset/configs/datasets/MyDataset.yaml
```

Minimal example for detection:

```yaml
# configs/datasets/MyDataset.yaml
path: /path/to/your/dataset   # or a relative path that resolves against datasets_dir
train: images/train
val: images/val
# test: images/test          # optional

names:
  0: person
  1: car
  2: bicycle
  # ...
```

For classification the notebook expects a folder layout instead of a YAML (see Ultralytics docs).  
For other tasks follow the official Ultralytics dataset format for that task.

**Important:**  
If you use a relative `path:`, make sure the notebook’s environment section points `datasets_dir` at the correct root, otherwise Ultralytics will look in the wrong place and try to re-download.

---

## Step 4 — Edit the configuration cells in the notebook

Open `notebook-v3.ipynb` (or `notebook.ipynb`) and go to **Section 1 — Experiment Setup**.

Change only these lines:

```python
TASK = "detect"                    # ← your task
MODEL = "yolo26n.pt"               # ← start with the nano model to validate the pipeline
DATA = "configs/datasets/MyDataset.yaml"
EXPERIMENT_NAME = "baseline"       # ← self-describing name, e.g. "0907_yolo26n_mydataset_e100"
PROJECT = f"runs/{TASK}"           # or a Platform project slug
```

Optionally create an overrides file:

```
configs/experiments/baseline.yaml
```

```yaml
# only put settings you have a strong reason to change
epochs: 100
imgsz: 640
batch: 16
# lr0: 0.01          # leave commented if the default is fine
```

The notebook automatically loads any existing file named after `EXPERIMENT_NAME`.

---

## Step 5 — Run the notebook top-to-bottom

Recommended first-pass order:

1. **Section 0** — install / check environment (`yolo checks`).
2. **Section 1** — confirm TASK / MODEL / DATA print correctly.  
   Watch for the model-suffix warning; fix any mismatch before continuing.
3. **Section 2** — run the structural check and the visual spot-check.  
   Set `RUN_SMOKE_TEST = True` the first time you use a new dataset (or after any label change).  
   Inspect `labels.jpg`, `labels_correlogram.jpg` and `train_batch0.jpg`.
4. **Section 3** — train. Start with the nano model and a modest number of epochs.
5. **Section 4** — evaluate `best.pt` and look at the plots and a few predictions.
6. **Section 5** — generate the summary (re-runnable even in a fresh kernel).
7. **Section 6** — export if you need a deployment format; re-validate the exported model.
8. **Section 7** — write the markdown report under `docs/`.

You should only write custom Python in the rare places the notebook itself marks as “customization points” (or under `scripts/`).

---

## Step 6 — Optional hyperparameter tuning (only after a healthy baseline)

Tuning is **off by default** (`RUN_TUNE = False` in Section 3). Follow this order:

1. Finish a normal single-shot training and inspect Section 4.
2. Only if the baseline looks sound, set `RUN_TUNE = True`, choose modest `TUNE_EPOCHS` /
   `TUNE_ITERATIONS`, and run the tuning cell.
3. Open `runs/<task>/<EXPERIMENT_NAME>_tune/best_hyperparameters.yaml`.
4. Promote the values you want into a **new** experiment YAML, pick a new
   `EXPERIMENT_NAME`, leave `RUN_TUNE = False`, and re-run a full-length training.

Do not treat the short-search run as the final model. See the notebook intro section
"Optional hyperparameter tuning (baseline-first workflow)" and the `yolo-tuning` skill.

---

## Step 7 — Keep the project tidy

- All Ultralytics outputs already live under `runs/`. Do not invent parallel folders.
- Put qualitative notes and next steps inside the generated report or in `docs/`.
- If you need a one-off conversion script, place it under `scripts/data/` and document it in the project README.
- Prefer Ultralytics converters (`ultralytics.data.converter`) before writing your own.

---

## Step 8 — Adding more experiments later

You do **not** create a new project folder for every model size or hyper-parameter change.

Instead:

1. Change `MODEL` or create a new overrides YAML.
2. Give it a new `EXPERIMENT_NAME` (e.g. `yolo26s_imgsz1280`).
3. Re-run the notebook (or just the training + evaluation sections).

All runs stay under the same project, making comparison trivial.

---

## Common mistakes to avoid

| Mistake | Why it hurts | Fix |
|---------|--------------|-----|
| Wrong TASK vs. model suffix | Silent failure → mAP stays near zero | Use the sanity-check cell in Section 1 |
| Relative dataset path + wrong `datasets_dir` | “Dataset not found” even though data is present | Set `datasets_dir` once per environment |
| Skipping the smoke test | Hours of training on broken labels | Run it once per new / changed dataset |
| Evaluating `last.pt` instead of `best.pt` | You may report an over-fitted checkpoint | Always load `weights/best.pt` |
| Putting every Ultralytics argument into the overrides YAML | Hides the real intentional changes | Only record settings you have a reason to change |
| Creating a new project for every model variant | Explodes the folder structure | Keep experiments inside one task+dataset project |
| Enabling `RUN_TUNE` before a healthy baseline | Wastes GPU on a broken pipeline | Inspect Section 4 first; promote hyps then retrain fully |

---

## Checklist before you call a project “done”

- [ ] Dataset YAML (or classification folder) is correct and checked.
- [ ] Visual spot-check and (at least one) smoke test passed.
- [ ] `TASK` and model suffix match.
- [ ] Experiment name is self-describing.
- [ ] Overrides file contains only intentional changes.
- [ ] If tuning was used: promoted best hyps into a new experiment and retrained fully.
- [ ] Evaluation used `best.pt`.
- [ ] A markdown report exists under `docs/`.
- [ ] You can re-open the notebook in a fresh session and still understand what was run.

---

You now know how to turn the template into a real project.  
For a deeper understanding of every cell, continue to `03_notebook_walkthrough.md`.
