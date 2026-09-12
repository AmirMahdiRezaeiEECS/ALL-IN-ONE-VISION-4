# Experiment Report — baseline

## Configuration
- Task: segment
- Model: yolo26n-seg.pt
- Data: carparts-seg.yaml
- Overrides: {}

## Training
- Run directory: `/content/runs/segment/runs/segment/baseline`
- Epochs ran: 101

## Optional tuning
- Ran: False
- (If True) best hyperparameters file: `runs/segment/baseline_tune/best_hyperparameters.yaml`
- Promote useful values into a new experiment YAML and retrain fully; do not ship the short-search run.

## Evaluation
- Final metrics: {'metrics/precision(B)': 0.6019985481728574, 'metrics/recall(B)': 0.823469109428278, 'metrics/mAP50(B)': 0.6794194120837568, 'metrics/mAP50-95(B)': 0.5687980641714278, 'metrics/precision(M)': 0.6037248893382141, 'metrics/recall(M)': 0.8230375855034263, 'metrics/mAP50(M)': 0.6861052144432808, 'metrics/mAP50-95(M)': 0.5498391791294871, 'fitness': 1.1186372433009149}

## Export
- Format: onnx
- Artifact: `/content/runs/segment/runs/segment/baseline/weights/best.onnx`
- Exported metrics: {'metrics/precision(B)': 0.5784848979563704, 'metrics/recall(B)': 0.8270525341863693, 'metrics/mAP50(B)': 0.6577433661345304, 'metrics/mAP50-95(B)': 0.5494312245791912, 'metrics/precision(M)': 0.580846863581176, 'metrics/recall(M)': 0.8291622484788806, 'metrics/mAP50(M)': 0.6653648280670125, 'metrics/mAP50-95(M)': 0.5364203192700078, 'fitness': 1.085851543849199}

## Notes
_Add qualitative observations, failure cases, and next steps here._
