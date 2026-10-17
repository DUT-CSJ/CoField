import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ImageDataset, SUNSeg
from evaluate import evaluate
from flow import training_loss
from model import build_model


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path):
    config = json.loads(Path(path).read_text())
    integer_keys = ["classes", "image_size", "epochs", "batch_size", "distance_radius",
                    "boundary_radius", "warmup_steps", "workers", "seed"]
    if config["mode"] == "sunseg":
        integer_keys += ["temporal_dim", "temporal_window"]
    for key in integer_keys:
        if not isinstance(config[key], int):
            raise ValueError(f"Set {key} to an integer in {path}")
    for key in ("data_root", "output_dir"):
        if not config[key] or "<" in config[key]:
            raise ValueError(f"Set {key} in {path}")
    if config["image_size"] % 16:
        raise ValueError("image_size must be divisible by 16")
    if min(config["epochs"], config["batch_size"], config["distance_scale"],
           config["image_size"], config["distance_radius"], config["classes"]) <= 0:
        raise ValueError("Training sizes, distance radius/scale and classes must be positive")
    if min(config["workers"], config["boundary_radius"], config["warmup_steps"],
           config["boundary_weight"], config["agreement_weight"]) < 0:
        raise ValueError("Worker counts, radii, warmup and loss weights cannot be negative")
    if config["source_range"][0] >= config["source_range"][1] or min(config["time_beta"]) <= 0:
        raise ValueError("Invalid source range or Beta parameters")
    return config


def make_scheduler(optimizer, total_steps, warmup):
    def scale(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min((step - warmup) / max(total_steps - warmup, 1), 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_checkpoint(path, model, config, epoch, fold, metrics):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": config, "epoch": epoch, "fold": fold, "metrics": metrics}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config["seed"] + args.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if config["mode"] == "sunseg":
        dataset = SUNSeg(config)
        validation = None
        print(f"SUN-SEG: {dataset.clips} training clips, {len(dataset)} target frames", flush=True)
    else:
        dataset = ImageDataset(config, "train", args.fold)
        validation = ImageDataset(config, "val", args.fold)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["workers"], drop_last=False,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    validation_loader = None if validation is None else DataLoader(
        validation, batch_size=1, shuffle=False, num_workers=config["workers"])
    initialization = config.get("init_checkpoint")
    if initialization and "<" in initialization:
        raise ValueError("Set init_checkpoint to a static checkpoint path or an empty string")
    model = build_model(config, pretrained=not bool(initialization)).to(device)
    if initialization:
        saved = torch.load(initialization, map_location="cpu")
        missing, unexpected = model.load_state_dict(saved["model"], strict=False)
        if unexpected or any(not key.startswith("temporal.") for key in missing):
            raise ValueError(f"Incompatible initialization: missing={missing}, unexpected={unexpected}")
    model.gradient_checkpointing = True
    encoder_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        group = encoder_parameters if name.startswith("image_encoder.") else other_parameters
        group.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 3e-5},
        {"params": other_parameters, "lr": 1e-4},
    ], weight_decay=config["weight_decay"])
    scheduler = make_scheduler(optimizer, config["epochs"] * len(loader), config["warmup_steps"])
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_fold{args.fold}" if config["mode"] == "image" else ""
    best = -1.0
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        components = {"flow": 0.0, "grid": 0.0, "agreement": 0.0}
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss, parts = training_loss(model, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            old_scale = scaler.get_scale()
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            count = batch["image"].shape[0]
            total += loss.detach().item() * count
            for key in components:
                components[key] += parts[key].item() * count
        metrics = evaluate(model, validation_loader, config, device) if validation_loader else None
        record = {"epoch": epoch, "loss": total / len(dataset),
                  "components": {key: value / len(dataset) for key, value in components.items()},
                  "metrics": metrics}
        print(json.dumps(record, allow_nan=False), flush=True)
        with (output / f"history{suffix}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        save_checkpoint(output / f"last{suffix}.pt", model, config, epoch, args.fold, metrics)
        if metrics is not None and metrics["dice"] is not None and metrics["dice"] > best:
            best = metrics["dice"]
            save_checkpoint(output / f"best{suffix}.pt", model, config, epoch, args.fold, metrics)


if __name__ == "__main__":
    main()
