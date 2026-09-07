# Start Here — Teaching Guide for ALL-IN-ONE-VISION-4 (A4)

Welcome. This documentation teaches you **what A4 is**, **how the pieces fit together**, and **how to turn the general template into a real project** of your own.

Read the files in this order:

| # | File | What you will learn |
|---|------|---------------------|
| 1 | `00_start_here.md` (this file) | Orientation and the big idea |
| 2 | `01_what_is_going_on.md` | Architecture, philosophy, and the notebook lifecycle |
| 3 | `02_how_to_edit_the_template.md` | Step-by-step recipe for creating a new project |
| 4 | `03_notebook_walkthrough.md` | Section-by-section explanation of `notebook-v3.ipynb` |

---

## Who this is for

You already know a little Python and have trained (or at least run) a model before.  
You do **not** need to be a Detectron2 expert or an Ultralytics internal.  
You want a **simple, reproducible way** to fine-tune YOLO models on your own data without reinventing training loops, logging, or export code.

---

## The one-sentence summary

**Every project in A4 is an edited copy of the same general template.**  
The template (especially `template/notebook-v3.ipynb`) is the primary product. Individual folders under `projects/` are just task- and dataset-specific instances of that template.

---

## Why this design exists

Most computer-vision experiments fail for two boring reasons:

1. A dataset problem that was never caught before training started.
2. An experiment that cannot be reproduced because nobody recorded what was actually run.

A4 is deliberately structured to make both mistakes hard. It does this by:

- Putting the entire experiment lifecycle in **one guided notebook**.
- Forcing configuration into small, intentional YAML files.
- Delegating all real ML work (train / val / predict / export) to **Ultralytics**.
- Keeping custom code rare and confined to clearly marked places.

---

## Two-minute mental model

```
template/                     ← the reusable blueprint
    ↓  (copy + edit)
projects/<Task>/<Dataset>/    ← a concrete experiment (e.g. Detect/VOC)
```

Inside every project you will find the same skeleton:

- `notebook.ipynb` (or `notebook-v3.ipynb`) — the single source of truth for the experiment
- `configs/datasets/` — dataset YAML(s)
- `configs/experiments/` — only the hyper-parameter overrides you have a reason to change
- `datasets/` — (optional) local data or notes about where data lives
- `runs/` — everything Ultralytics writes (weights, plots, args.yaml, results.csv …)
- `docs/` — experiment reports and notes
- `scripts/` — rare one-off helpers (data conversion, custom inference)

You almost never invent a new structure. You only fill in the blanks that the template already prepared for you.

---

## Next step

Open `01_what_is_going_on.md` to understand the design principles and the notebook’s seven stages.  
Then go to `02_how_to_edit_the_template.md` when you are ready to create your first project.
