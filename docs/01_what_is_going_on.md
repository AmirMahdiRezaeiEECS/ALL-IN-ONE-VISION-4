# What Is Going On — Architecture & Philosophy

This document explains the **why** behind A4 so the later “how-to” steps make sense.

---

## 1. Core Principles (from the repository README)

A4 follows three deliberate rules:

### 1. Good-enough, bottleneck-driven, extremely low-code

The goal is **not** a massive new framework.  
The goal is a simple, practical workflow that removes the real bottlenecks people hit when fine-tuning YOLO:

- forgetting to validate the dataset,
- losing track of what was actually trained,
- rewriting the same training / export boilerplate every time.

If Ultralytics already solves something well, A4 uses it instead of re-implementing it.

### 2. Build on mature software

Ultralytics already owns:

- model loading,
- training loops,
- validation metrics & plots,
- prediction / visualization,
- export to 20+ formats,
- the built-in hyper-parameter tuner.

A4 therefore concentrates on the thin layer **around** Ultralytics: experiment setup, dataset sanity checks, reproducible summaries, and reporting.

### 3. Standard first, zero reinvention

Before writing any custom code, check:

- [Ultralytics Docs](https://docs.ultralytics.com/)
- Official notebooks and the Platform
- The `yolo-*` skills that live in this repository

Only when those are insufficient do you write a few lines of your own (and you put them in the clearly marked places).

---

## 2. Project Template as the Primary Product

```
ALL-IN-ONE-VISION-4/
├── template/                 ← the general, reusable blueprint (primary product)
│   ├── configs/
│   │   ├── datasets/
│   │   └── experiments/
│   ├── datasets/
│   ├── docs/
│   ├── notebook-v3.ipynb     ← the heart of every experiment
│   ├── runs/
│   ├── scripts/
│   └── README.md
│
├── projects/                 ← concrete instances of the template
│   ├── Detect/
│   │   └── VOC/
│   ├── Segment/
│   ├── Pose/
│   ├── OBB/
│   ├── Classify/
│   ├── Depth/
│   └── Semantic/
│
└── docs/                     ← repository-level teaching material (you are here)
```

**Key idea:**  
You never start a new project from an empty folder.  
You copy the template and then change only the parts that must be different for your task + dataset.

Organization is **by task and dataset**, not by model.  
One project folder can contain many experiments (different model sizes, different overrides, different experiment names). You do **not** create a new top-level project for every model variant.

---

## 3. Notebook-Centered Design

Every project is driven by a **single Jupyter notebook** that acts as:

- the experiment interface,
- the workflow documentation,
- the place where configuration is declared,
- the place where results are inspected and summarized.

The notebook is **not** a collection of independent examples.  
It is a guided, top-to-bottom executable experiment with seven clear stages:

| Stage | Name | Purpose |
|-------|------|---------|
| 0 | Environment | Install / verify Ultralytics, GPU, paths |
| 1 | Experiment Setup | Declare TASK, MODEL, DATA, experiment name, overrides |
| 2 | Dataset Preparation & Validation | Structure check → visual spot-check → smoke test |
| 3 | Model & Training | Load pretrained weights, train, optional tuning |
| 4 | Evaluation & Inspection | Evaluate `best.pt`, read the plots, look at predictions |
| 5 | Experiment Summary | Assemble config + metrics into one readable record |
| 6 | Export | Convert to deployment format and re-validate |
| 7 | Report | Write a durable markdown report under `docs/` |

Each stage contains markdown that explains **why** the step exists and **what goes wrong** if you skip it. That is intentional teaching material, not decoration.

---

## 4. Default-First Configuration

Ultralytics ships good defaults.  
A4 therefore treats configuration as **overrides only**.

- Put a dataset definition in `configs/datasets/<name>.yaml`.
- Put only the settings you have a strong reason to change in `configs/experiments/<experiment_name>.yaml`.
- Leave everything else to Ultralytics.
- A `class_aliases` mapping (Section 3 of the notebook, for reusing pretrained classification-head weights when a checkpoint and your dataset name a shared class differently) is another example of an intentional override, alongside hyperparameters — add it only after finding a real mismatch, not by default.

This keeps every deviation from the baseline explicit, diffable, and easy to understand six months later.

---

## 5. What lives where (mental map)

| Location | Responsibility |
|----------|----------------|
| `notebook-v3.ipynb` | Workflow, configuration points, inspection, summary, report generation |
| `configs/datasets/` | Dataset YAML (paths, class names, splits) |
| `configs/experiments/` | Intentional hyper-parameter overrides |
| `runs/` | All Ultralytics outputs (weights, plots, args.yaml, results.csv …) |
| `docs/` (inside a project) | Experiment reports and qualitative notes |
| `scripts/data/` | Rare one-off conversion scripts (prefer Ultralytics converters first) |
| `scripts/inference/` | Rare custom inference helpers |

You almost never need a separate `artifacts/`, `checkpoints/`, or `logs/` folder — Ultralytics already manages that under `runs/`.

---

## 6. The two failure modes the notebook is designed to prevent

1. **Dataset problems that only appear after hours of training**  
   → Stages 2 (structure check → visual check → smoke test) catch them early.

2. **Unreproducible experiments**  
   → Stages 1, 5 and 7 force you to name the experiment, record overrides, and write a report that lives on disk.

Understanding these two motivations makes every cell in the notebook feel purposeful instead of arbitrary.

---

Next: `02_how_to_edit_the_template.md` — the concrete recipe for creating a new project.
