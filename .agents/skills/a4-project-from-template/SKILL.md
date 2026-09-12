---
name: a4-project-from-template
description: >
  Use when creating a new A4 project from the general template, turning template/
  into projects/Task/Dataset, adapting notebook-v3.ipynb for a task and dataset,
  or scaffolding Detect/Segment/Pose/OBB/Classify/Depth/Semantic experiments.
  Triggers include new project, from template, copy template, create project folder.
---

# Create an A4 Project from the Template

Every project under `projects/` is an edited copy of `template/`. Never start from an empty folder. The template (especially `template/notebook-v3.ipynb`) is the primary product; projects are task+dataset instances of it.

## When to use this skill

- User asks to create / scaffold / start a new project for a dataset or task
- User wants to turn the template into `projects/<Task>/<Dataset>/`
- User needs the exact steps to adapt configs, notebook cells, and structure

For pure Ultralytics training/dataset/export details, also load the matching `yolo-*` skill.

## Core rules (do not violate)

1. **Copy the whole template tree** — keep structure identical.
2. **Organize by task + dataset**, not by model. One project holds many experiments.
3. **Edit only designated places** — TASK/MODEL/DATA/EXPERIMENT_NAME cells, dataset YAML, optional overrides YAML.
4. **Minimal overrides** — put only intentional changes in `configs/experiments/`. Leave everything else to Ultralytics defaults.
5. **Validate before train** — structural check + visual smoke test (`RUN_SMOKE_TEST=True` first time).
6. **Always evaluate `best.pt`**, never `last.pt`.
7. **Custom code is rare** — only in marked notebook cells or under `scripts/`. Prefer Ultralytics converters and APIs.

## Recipe

### 1. Choose names

- **Task** (must match notebook vocabulary): `detect` | `segment` | `semantic` | `depth` | `classify` | `pose` | `obb`
- **Dataset** short name: e.g. `VOC`, `MyCars`, `WarehouseBoxes`
- Target path: `projects/<TaskCapitalized>/<Dataset>/`  
  Example: detection on VOC → `projects/Detect/VOC/`

### 2. Copy the template

From repository root:

```bash
cp -r template projects/Detect/MyDataset
```

Expected layout after copy:

```
projects/Detect/MyDataset/
├── configs/
│   ├── datasets/
│   └── experiments/
├── datasets/
├── docs/
├── notebook-v3.ipynb      # rename to notebook.ipynb if preferred
├── runs/
├── scripts/
└── README.md
```

### 3. Dataset config

Create `configs/datasets/<Dataset>.yaml` (detection-style example):

```yaml
path: /absolute/or/relative/path   # relative paths resolve against datasets_dir
train: images/train
val: images/val
# test: images/test

names:
  0: person
  1: car
  # ...
```

- Classification usually uses a folder layout (see Ultralytics docs), not this YAML shape.
- Other tasks follow the official Ultralytics dataset format for that task.
- If `path:` is relative, ensure the notebook environment sets `datasets_dir` correctly.

### 4. Notebook Section 1 — Experiment Setup

Edit only these identity cells:

```python
TASK = "detect"                    # must match model suffix
MODEL = "yolo26n.pt"               # start nano to validate the pipeline
DATA = "configs/datasets/MyDataset.yaml"
EXPERIMENT_NAME = "baseline"       # self-describing, e.g. "0907_yolo26n_mydataset_e100"
PROJECT = f"runs/{TASK}"           # or Platform project slug
```

Optional overrides file `configs/experiments/<EXPERIMENT_NAME>.yaml`:

```yaml
# only intentional changes
epochs: 100
imgsz: 640
batch: 16
```

The notebook loads the overrides file automatically when it exists and matches `EXPERIMENT_NAME`.

**Optional — class-name aliasing.** The same overrides file can also carry a
`class_aliases` key (source/pretrained class name → this dataset's class name).
It is consumed by Section 3, not `model.train()` directly, and lets Ultralytics'
name-based `cls_remap` transfer pretrained classification-head rows for classes
the checkpoint and the dataset name differently (e.g. COCO `airplane` vs. VOC
`aeroplane`):

```yaml
# configs/experiments/baseline_aliased.yaml
epochs: 100
imgsz: 640
batch: 16

class_aliases:
  airplane: aeroplane
  motorcycle: motorbike
```

Only add this **after** a `baseline` run with no aliases, and only for name
mismatches you actually found (Section 2's discovery cell, or by eye) — don't
guess. See the notebook's "Optional — Transfer Classes with Name Aliases"
intro section and Section 3 for the full mechanics.

**`MODEL` must stay the untouched, officially pretrained checkpoint** for the
aliased run — do not repoint it at the `baseline` run's `best.pt`. Aliasing
renames names on the *original* pretrained classification head so its weights
transfer; a fine-tuned checkpoint's names are already the dataset's names (so
no `class_aliases` key matches) and its weights are already dataset-specific,
not the general pretrained ones the technique is meant to preserve. Only
`EXPERIMENT_NAME` (and the overrides file) should differ between the baseline
and aliased runs — `MODEL` stays identical.

### 5. Run order (first pass)

1. **§0 Environment** — install, `yolo checks`, optional Platform key / `datasets_dir`.
2. **§1 Setup** — confirm TASK / MODEL / DATA print; fix model-suffix mismatch warnings.
3. **§2 Dataset** — structural check + visual spot-check. Set `RUN_SMOKE_TEST = True` for new/changed data. Inspect `labels.jpg`, `labels_correlogram.jpg`, `train_batch0.jpg`. Optionally set `SHOW_PRETRAINED_CLASSES = True` to list the checkpoint's class names alongside the dataset's, for spotting aliasing candidates.
4. **§3 Train** — start with nano + modest epochs. Leave `class_aliases` unset for this first (`baseline`) run.
5. **§4 Eval** — load `weights/best.pt`, inspect metrics and predictions.
6. **§5 Summary** — re-runnable even in a fresh kernel.
7. **§6 Export** — only if needed; re-validate the exported artifact.
8. **§7 Report** — write markdown under `docs/`.

Write custom Python only in cells the notebook marks as customization points, or place helpers under `scripts/data/` / `scripts/inference/`.

### 6. Later experiments (same project)

Do **not** create a new project folder for model size, hyper-parameter, or class-aliasing changes.

1. Change `MODEL` and/or create a new overrides YAML.
2. Set a new self-describing `EXPERIMENT_NAME`.
3. Re-run the relevant sections (train + eval + summary).

All runs stay under the same `projects/<Task>/<Dataset>/runs/`.

**Class-aliasing follows the same baseline-first pattern as hyperparameter
tuning**: run `baseline` (no aliases) first, then a separate `baseline_aliased`
(or similarly named) experiment with `class_aliases` set, then compare
per-class metrics between the two runs' confusion matrices before deciding
which one to keep. Never treat an aliased run as a straight replacement for the
baseline without that comparison — aliasing a wrong pair of classes can quietly
hurt both.

**Exception to step 1 above:** for a class-aliasing experiment specifically, do
**not** change `MODEL` to the previous run's checkpoint — leave it as the same
official pretrained weights used for `baseline`. Only add the `class_aliases`
overrides file and a new `EXPERIMENT_NAME`.

## Checklist before calling the project done

- [ ] Dataset YAML (or classification folders) correct and verified
- [ ] Visual spot-check + at least one smoke test passed
- [ ] `TASK` matches model suffix
- [ ] Experiment name is self-describing
- [ ] Overrides contain only intentional changes
- [ ] Evaluation used `best.pt`
- [ ] If class aliasing was used: an unaliased `baseline` exists and per-class metrics were compared against it
- [ ] Markdown report exists under `docs/`
- [ ] Notebook can be re-opened in a fresh session and still understood

## Common failure modes

| Mistake | Fix |
|---------|-----|
| Wrong TASK vs. model suffix | Use the sanity-check cell in §1 |
| Relative `path:` + wrong `datasets_dir` | Set `datasets_dir` once per environment |
| Skipped smoke test | Run it once per new/changed dataset |
| Evaluating `last.pt` | Always load `weights/best.pt` |
| Full Ultralytics arg dump in overrides | Record only intentional changes |
| New project per model variant | Keep experiments inside one task+dataset project |
| Guessing `class_aliases` without checking per-class metrics | Compare aliased vs. baseline confusion matrices; keep the aliased run only if it helps |

## Relationship to other skills

- After scaffolding, use `yolo-datasets` for label formats, conversion, and validation details.
- Use `yolo-training` / `yolo-tuning` for train args, recipes, and hyper-parameter search.
- Use `yolo-models` for weight choice, `yolo-inference` for predict/track, `yolo-export` for deployment formats.
- The top-level `yolo` skill routes across the lifecycle.

This skill owns the **A4 project creation and template-adaptation workflow**. The `yolo-*` skills own the underlying Ultralytics mechanics.
