from collections import deque

import torch
import torch.nn.functional as F


def distance_to_set(seed, radius):
    reached = seed.bool()
    distance = torch.full_like(seed, float(radius))
    distance = distance.masked_fill(reached, 0)
    for step in range(1, radius + 1):
        reached = F.max_pool2d(reached.float(), 3, 1, 1).bool()
        distance = torch.where(reached & (distance > step), float(step), distance)
    return distance


def structured_endpoint(labels, classes, config):
    indices = torch.arange(1, classes + 1, device=labels.device)[None, :, None, None]
    masks = (labels[:, None] == indices).float()
    radius = int(config["distance_radius"])
    distance = distance_to_set(1 - masks, radius) - distance_to_set(masks, radius)
    distance = torch.tanh(distance / config["distance_scale"])
    band_radius = int(config["boundary_radius"])
    kernel = 2 * band_radius + 1
    dilation = F.max_pool2d(masks, kernel, 1, band_radius)
    erosion = 1 - F.max_pool2d(1 - masks, kernel, 1, band_radius)
    band = (dilation - erosion).clamp(0, 1)
    endpoint = torch.cat((2 * masks - 1, distance, 2 * band - 1), 1)
    return endpoint, band, masks.sum((2, 3)) > 0


def endpoint_agreement(model, condition, source, present):
    condition = tuple(value.detach() if value is not None else None for value in condition)
    _, first = model.decode_state(condition, source, 1.0)
    one_step = source - first.float()
    midpoint = (source - 0.5 * first.float()).detach()
    with torch.no_grad():
        _, second = model.decode_state(condition, midpoint, 0.5)
        reference = (midpoint - 0.5 * second.float()).detach()
    classes = model.classes
    occupancy = (torch.sigmoid(4 * one_step[:, :classes])
                 - torch.sigmoid(4 * reference[:, :classes])).abs().mean()
    boundary = (torch.sigmoid(4 * one_step[:, 2 * classes:])
                - torch.sigmoid(4 * reference[:, 2 * classes:])).abs().mean()
    differences = (one_step[:, classes:2 * classes] - reference[:, classes:2 * classes]).abs().mean((2, 3))
    distance = (differences * present).sum() / present.sum().clamp_min(1)
    return occupancy + 0.5 * distance + 0.25 * boundary


def training_loss(model, batch, config):
    images = batch["image"].float() * 2 - 1
    labels = batch["mask"]
    endpoint, bands, present = structured_endpoint(labels, model.classes, config)
    source = torch.empty_like(endpoint).uniform_(*config["source_range"])
    alpha, beta = config["time_beta"]
    concentration_a = images.new_tensor(alpha)
    concentration_b = images.new_tensor(beta)
    timestep = torch.distributions.Beta(concentration_a, concentration_b).sample((images.shape[0],))
    state = endpoint.lerp(source, timestep[:, None, None, None])
    target = source - endpoint
    history = None
    if model.temporal is not None and batch["history_valid"].any():
        with torch.no_grad():
            past = model.encode_image(batch["previous"].float() * 2 - 1)
        history = [past.detach()]
    condition = model.encode_condition(images, history, batch.get("history_valid"))
    grid, velocity = model.decode_state(condition, state, timestep)
    weight = 1 + config["boundary_weight"] * bands.repeat(1, 3, 1, 1)
    flow = (weight * (velocity.float() - target).square()).sum() / weight.sum()
    grid_target = F.interpolate(target, grid.shape[-2:], mode="bilinear", align_corners=False)
    grid_loss = F.mse_loss(grid.float(), grid_target)
    agreement = endpoint_agreement(model, condition, source, present) if config["agreement_weight"] else flow.new_zeros(())
    loss = flow + 0.1 * grid_loss + config["agreement_weight"] * agreement
    return loss, {"flow": flow.detach(), "grid": grid_loss.detach(), "agreement": agreement.detach()}


def decode_mask(endpoint, size, classes):
    occupancy = F.interpolate(endpoint[:, :classes].float(), size=size,
                              mode="bilinear", align_corners=False)
    if classes == 1:
        return (occupancy[:, 0] > 0).long()
    score, label = occupancy.max(1)
    return torch.where(score > 0, label + 1, 0)


class Predictor:
    def __init__(self, model, source_range=(-0.05, 0.05), steps=1, seed=2025):
        if steps < 1:
            raise ValueError("steps must be positive")
        self.model = model
        self.source_range = source_range
        self.steps = steps
        self.generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
        self.history = deque(maxlen=2)
        self.sequence = None

    def reset(self, sequence=None):
        self.history.clear()
        self.sequence = sequence

    @torch.no_grad()
    def __call__(self, images, sequence=None):
        if self.model.temporal is not None:
            if images.shape[0] != 1 or sequence is None:
                raise ValueError("Video inference requires one frame and a sequence identifier")
            if sequence != self.sequence:
                self.reset(sequence)
        images = images.float() * 2 - 1
        condition = self.model.encode_condition(images, list(self.history))
        state = torch.empty(images.shape[0], self.model.channels, *images.shape[-2:],
                            device=images.device).uniform_(*self.source_range, generator=self.generator)
        for step in range(self.steps):
            time = 1 - step / self.steps
            _, velocity = self.model.decode_state(condition, state, time)
            state = state - velocity.float() / self.steps
        if self.model.temporal is not None:
            self.history.append(condition[0].detach())
        return state
