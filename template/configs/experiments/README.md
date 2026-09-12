# Experiment overrides

One YAML file per experiment, named after `EXPERIMENT_NAME` in the notebook:
`<EXPERIMENT_NAME>.yaml`. The notebook loads it automatically in Section 1 if it
exists and passes its contents to `model.train()` (Section 3).

Only put **intentional deviations** from Ultralytics' defaults here — this file
is a record of what and why, not a full config dump. If a setting doesn't need
to change, leave it out and let Ultralytics use its default.

```yaml
# configs/experiments/baseline.yaml
epochs: 100
imgsz: 640
batch: 16
```

## `class_aliases` (optional, special-cased)

Unlike every other key, `class_aliases` is **not** passed to `model.train()` —
the notebook pops it out in Section 3 and uses it to rename the pretrained
checkpoint's class names in memory before training, so Ultralytics' name-based
`cls_remap` can transfer classification-head weights for classes the checkpoint
and this dataset name differently.

```yaml
# configs/experiments/baseline_aliased.yaml
epochs: 100
imgsz: 640
batch: 16

class_aliases:
  airplane: aeroplane     # pretrained (source) name: this dataset's name
  motorcycle: motorbike
```

Only add this after training an unaliased `baseline` first and finding a real
name mismatch — see the notebook's "Optional — Transfer Classes with Name
Aliases" section for the full baseline-first workflow.

`MODEL` (in the notebook, Section 1) must stay the untouched, officially
pretrained checkpoint for both the baseline and the aliased run — never point
it at the baseline's own `weights/best.pt`. A fine-tuned checkpoint's class
names and weights are already dataset-specific, so aliasing it matches nothing
and preserves nothing.
