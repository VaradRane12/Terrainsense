from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os

MODELS = [
    ("COCO pre-trained baseline", "yolov8n.pt"),
    ("Retrain 1", "/Users/varad/Documents/Programming /Terrainsense/runs/detect/train6/weights/last.pt"),
    ("Retrain 2", "/Users/varad/Documents/Programming /Terrainsense/runs/detect/train10/weights/best.pt"),
]

OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

def evaluate_models():
    summary = {}
    per_class = {}
    for name, path in MODELS:
        print(f"Evaluating {name} -> {path}")
        m = YOLO(path)
        r = m.val(data='terrain_dataset/data.yaml', imgsz=640, conf=0.001, plots=False, verbose=False)
        box = r.box
        # overall
        mAP50 = float(r.results_dict.get('metrics/mAP50(B)', np.nan))
        mAP5095 = float(r.results_dict.get('metrics/mAP50-95(B)', np.nan))
        inference_ms = float(r.speed['inference'])
        summary[name] = {'mAP50': mAP50, 'mAP50-95': mAP5095, 'inference_ms': inference_ms}

        # per-class
        names = [r.names[i] for i in sorted(r.names.keys())]
        all_ap = np.array(box.all_ap) if hasattr(box, 'all_ap') else None
        ap50 = all_ap[:, 0].tolist() if all_ap is not None else (list(np.array(box.ap50)) if hasattr(box, 'ap50') else [])
        ap5095 = np.mean(all_ap, axis=1).tolist() if all_ap is not None else (list(np.array(box.ap)) if hasattr(box, 'ap') else [])
        recall = list(np.array(box.r)) if hasattr(box, 'r') else []
        cls = []
        for i, nm in enumerate(names):
            cls.append({'class': nm,
                        'AP50': ap50[i] if i < len(ap50) else None,
                        'AP50-95': ap5095[i] if i < len(ap5095) else None,
                        'Recall': recall[i] if i < len(recall) else None})
        per_class[name] = cls

    return summary, per_class

def plot_overall(summary):
    labels = list(summary.keys())
    mAP50 = [summary[l]['mAP50'] for l in labels]
    mAP5095 = [summary[l]['mAP50-95'] for l in labels]
    inference = [summary[l]['inference_ms'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x - width/2, mAP50, width, label='mAP50')
    ax.bar(x + width/2, mAP5095, width, label='mAP50-95')
    ax.set_ylabel('AP')
    ax.set_title('Model overall AP on validation set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'overall_AP.png')
    fig.savefig(out)
    print('Wrote', out)

    # inference times
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(labels, inference, color='tab:orange')
    ax.set_ylabel('Inference time (ms)')
    ax.set_title('Per-image inference time')
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, 'inference_ms.png')
    fig.savefig(out2)
    print('Wrote', out2)

def plot_per_class(per_class, model_name):
    cls = per_class[model_name]
    labels = [c['class'] for c in cls]
    ap50 = [c['AP50'] if c['AP50'] is not None else 0.0 for c in cls]
    ap5095 = [c['AP50-95'] if c['AP50-95'] is not None else 0.0 for c in cls]
    recall = [c['Recall'] if c['Recall'] is not None else 0.0 for c in cls]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(x - width, ap50, width, label='AP50')
    ax.bar(x, ap5095, width, label='AP50-95')
    ax.bar(x + width, recall, width, label='Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-class performance ({model_name})')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'per_class_{model_name}.png')
    fig.savefig(out)
    print('Wrote', out)

def main():
    summary, per_class = evaluate_models()
    plot_overall(summary)
    # pick best model by mAP50
    best = max(summary.keys(), key=lambda k: summary[k]['mAP50'])
    plot_per_class(per_class, best)
    print('Best model by mAP50:', best)

if __name__ == '__main__':
    main()
