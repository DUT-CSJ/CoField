import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
import torch
from torch.utils.data import DataLoader

from data import ImageDataset, SUNSeg, SUN_SUBSETS
from flow import Predictor, decode_mask
from model import build_model


class Metrics:
    def __init__(self, classes):
        self.values = [[] for _ in range(classes)]

    def update(self, prediction, reference):
        prediction = prediction.detach().cpu().numpy()
        reference = reference.detach().cpu().numpy()
        for index, values in enumerate(self.values, 1):
            pred, ref = prediction == index, reference == index
            if not pred.any() and not ref.any():
                continue
            intersection = (pred & ref).sum()
            dice = 2 * intersection / (pred.sum() + ref.sum())
            iou = intersection / (pred | ref).sum()
            hd95, assd = np.nan, np.nan
            if pred.any() and ref.any():
                structure = generate_binary_structure(2, 1)
                pred_border = pred & ~binary_erosion(pred, structure=structure)
                ref_border = ref & ~binary_erosion(ref, structure=structure)
                first = distance_transform_edt(~ref_border)[pred_border]
                second = distance_transform_edt(~pred_border)[ref_border]
                hd95 = max(np.percentile(first, 95), np.percentile(second, 95))
                assd = (first.sum() + second.sum()) / (len(first) + len(second))
            values.append((dice * 100, iou * 100, hd95, assd))

    def summary(self):
        names = ("dice", "iou", "hd95", "assd")
        per_class = []
        for index, values in enumerate(self.values, 1):
            array = np.asarray(values, dtype=float).reshape(-1, 4)
            row = {"class": index, "cases": len(values)}
            for column, name in enumerate(names):
                finite = array[:, column][np.isfinite(array[:, column])]
                row[name] = float(finite.mean()) if len(finite) else None
            row["undefined_surface_cases"] = int(np.isnan(array[:, 2]).sum())
            per_class.append(row)
        result = {}
        for name in names:
            valid = [row[name] for row in per_class if row[name] is not None]
            result[name] = float(np.mean(valid)) if valid else None
        result["per_class"] = per_class
        return result


@torch.no_grad()
def evaluate(model, loader, config, device, steps=1, save_dir=None):
    model.eval()
    predictor = Predictor(model, config["source_range"], steps, config["seed"])
    metrics = Metrics(model.classes)
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["mask"][0]
        endpoint = predictor(images, batch["sequence"][0])
        prediction = decode_mask(endpoint, labels.shape, model.classes)[0]
        metrics.update(prediction, labels)
        if save_dir is not None:
            path = Path(save_dir) / Path(batch["name"][0]).with_suffix(".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            dtype = np.uint8 if model.classes <= 255 else np.uint16
            Image.fromarray(prediction.cpu().numpy().astype(dtype)).save(path)
    return metrics.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--split", default="val")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--subset", choices=tuple(SUN_SUBSETS))
    parser.add_argument("--steps", type=int, choices=(1, 2, 5), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu")
    config = dict(saved["config"], data_root=args.data_root)
    if args.split_file:
        config["split_file"] = args.split_file
    model = build_model(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    results = {}
    subsets = [args.subset] if args.subset else list(SUN_SUBSETS)
    if config["mode"] == "image":
        subsets = [args.split]
    for subset in subsets:
        if config["mode"] == "image":
            fold = saved.get("fold", 0) if args.fold is None else args.fold
            dataset = ImageDataset(config, subset, fold)
        else:
            dataset = SUNSeg(config, subset)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        results[subset] = evaluate(model, loader, config, device, args.steps, args.save_dir)
    text = json.dumps(results, indent=2, allow_nan=False)
    print(text)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n")


if __name__ == "__main__":
    main()
