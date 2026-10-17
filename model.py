import math

import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, t):
        frequencies = torch.exp(-math.log(10) * torch.arange(128, device=t.device) / 128)
        phase = t.float()[:, None] * frequencies[None]
        return self.mlp(torch.cat((phase.cos(), phase.sin()), -1))


def patch_positions(height, width, device, dtype):
    row, column = torch.meshgrid(torch.arange(height, device=device),
                                 torch.arange(width, device=device), indexing="ij")
    integer = torch.stack((row.flatten(), column.flatten()), -1)
    position = integer.to(dtype)
    scale = position.new_tensor((max(height - 1, 1), max(width - 1, 1)))
    return position / scale * 2 - 1, integer


class RelationAttention(nn.Module):
    def __init__(self, width=384, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.position_bias = nn.Linear(2, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                 nn.Linear(4 * width, width))

    def forward(self, tokens, mask, relative, gates, patches):
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
        bias = self.position_bias(relative).permute(2, 0, 1)
        logits[:, :, :patches, :patches] += bias[None].float()
        logits[:, :, :patches, :patches] += gates.float().clamp_min(1e-6).log()[:, None]
        logits = logits.masked_fill(~mask[:, None], -torch.inf)
        attention = logits.softmax(-1).to(v.dtype)
        message = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.projection(message)
        return tokens + self.ffn(self.norm2(tokens))


class FieldWeaver(nn.Module):
    def __init__(self, channels, width=384, semantic_neighbors=8, anchors=4):
        super().__init__()
        summary_width = 4 * channels
        self.semantic_neighbors = semantic_neighbors
        self.state_projection = nn.Sequential(nn.LayerNorm(summary_width),
            nn.Linear(summary_width, width), nn.SiLU(), nn.Linear(width, width))
        self.position_projection = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.time_projection = nn.Linear(width, width)
        self.gate_projection = nn.Sequential(nn.LayerNorm(summary_width), nn.Linear(summary_width, 12), nn.SiLU())
        self.gate = nn.Sequential(nn.Linear(38, 32), nn.SiLU(), nn.Linear(32, 1))
        self.anchors = nn.Parameter(torch.randn(1, anchors, width) * 0.02)
        self.layers = nn.ModuleList([RelationAttention(width) for _ in range(2)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def summarize(state, grid):
        mean = F.adaptive_avg_pool2d(state, grid)
        variance = F.adaptive_avg_pool2d(state.square(), grid) - mean.square()
        std = variance.clamp_min(0).add(1e-8).sqrt()
        minimum = -F.adaptive_max_pool2d(-state, grid)
        maximum = F.adaptive_max_pool2d(state, grid)
        return torch.cat((mean, std, minimum, maximum), 1).flatten(2).transpose(1, 2)

    def topology(self, interaction, integer):
        batch, patches, _ = interaction.shape
        difference = integer[:, None] - integer[None]
        local = difference.abs().amax(-1) <= 1
        mask = local[None].expand(batch, -1, -1).clone()
        count = min(self.semantic_neighbors, patches - 1)
        if count:
            normalized = F.normalize(interaction.detach().float(), dim=-1)
            similarity = normalized @ normalized.transpose(1, 2)
            identity = torch.eye(patches, device=interaction.device, dtype=torch.bool)
            indices = similarity.masked_fill(identity[None], -torch.inf).topk(count, -1).indices
            mask.scatter_(2, indices, True)
        length = patches + self.anchors.shape[1]
        full = torch.ones(batch, length, length, dtype=torch.bool, device=interaction.device)
        full[:, :patches, :patches] = mask
        return full

    def forward(self, image, state, time, context, temporal=None):
        batch, patches, width = image.shape
        grid = (state.shape[-2] // 16, state.shape[-1] // 16)
        if patches != grid[0] * grid[1]:
            raise ValueError("Image and state patch grids differ")
        summary = self.summarize(state.float(), grid).to(image.dtype)
        position, integer = patch_positions(*grid, image.device, image.dtype)
        interaction = image + self.state_projection(summary)
        interaction = interaction + self.position_projection(position)[None]
        interaction = interaction + self.time_projection(time)[:, None] + context
        if temporal is not None:
            interaction = interaction + temporal
        relative = position[:, None] - position[None]
        gate_codes = self.gate_projection(summary)
        left = gate_codes[:, :, None].expand(-1, -1, patches, -1)
        right = gate_codes[:, None].expand(-1, patches, -1, -1)
        offsets = relative[None].expand(batch, -1, -1, -1)
        gates = self.gate(torch.cat((left, right, left - right, offsets), -1)).squeeze(-1).sigmoid()
        mask = self.topology(interaction, integer)
        anchors = self.anchors.expand(batch, -1, -1).to(interaction.dtype)
        tokens = torch.cat((interaction, anchors), 1)
        for layer in self.layers:
            tokens = layer(tokens, mask, relative, gates, patches)
        return image + self.output(self.output_norm(tokens[:, :patches]))


class LocalFieldBlock(nn.Module):
    def __init__(self, code_width=384, width=64):
        super().__init__()
        self.width = width
        self.parameters_from_code = nn.Linear(code_width, 4 * width * width)
        self.norm = RMSNorm(width)

    def forward(self, pixels, codes):
        width = self.width
        first, second = self.parameters_from_code(codes).chunk(2, -1)
        first = F.normalize(first.reshape(-1, width, 2 * width), dim=1)
        second = F.normalize(second.reshape(-1, 2 * width, width), dim=1)
        return pixels + torch.bmm(F.silu(torch.bmm(self.norm(pixels), first)), second)


class LocalFieldDecoder(nn.Module):
    def __init__(self, channels, code_width=384, width=64):
        super().__init__()
        axis = torch.linspace(0, 1, 16)
        row, column = torch.meshgrid(axis, axis, indexing="ij")
        frequencies = torch.linspace(0, 8, 8)
        fx = frequencies.reshape(1, 8, 1)
        fy = frequencies.reshape(1, 1, 8)
        coordinates = (torch.cos(torch.pi * column.flatten()[:, None, None] * fx)
                       * torch.cos(torch.pi * row.flatten()[:, None, None] * fy) / (1 + fx * fy))
        self.register_buffer("coordinates", coordinates.reshape(1, 256, 64), persistent=False)
        self.channels = channels
        self.input = nn.Linear(channels + 64, width)
        self.blocks = nn.ModuleList([LocalFieldBlock(code_width, width) for _ in range(2)])
        self.modulation = nn.Linear(code_width, 2 * width)
        self.norm = RMSNorm(width)
        self.output = nn.Linear(width, channels)
        for layer in (self.modulation, self.output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, codes, state):
        batch, channels, height, width = state.shape
        patches = F.unfold(state, 16, stride=16).transpose(1, 2)
        count = patches.shape[1]
        pixels = patches.reshape(batch * count, channels, 256).transpose(1, 2)
        positions = self.coordinates.to(pixels.dtype).expand(batch * count, -1, -1)
        pixels = self.input(torch.cat((pixels, positions), -1))
        codes = codes.reshape(batch * count, -1)
        gamma, delta = self.modulation(codes).chunk(2, -1)
        for block in self.blocks:
            pixels = block(pixels, codes) * (1 + gamma[:, None]) + delta[:, None]
        pixels = self.output(self.norm(pixels))
        tiles = pixels.transpose(1, 2).reshape(batch, count, channels * 256)
        return F.fold(tiles.transpose(1, 2), (height, width), 16, stride=16)


class TemporalRouteAdapter(nn.Module):
    def __init__(self, dimension, window, width=384):
        super().__init__()
        if dimension < 1 or window < 1 or window % 2 == 0:
            raise ValueError("Temporal dimension must be positive and window must be positive and odd")
        self.window = window
        self.dimension = dimension
        self.query = nn.Linear(width, dimension)
        self.key = nn.Linear(width, dimension)
        self.value = nn.Linear(width, dimension)
        self.relative_bias = nn.Linear(3, 1)
        self.gate = nn.Sequential(nn.Linear(width + dimension, dimension), nn.SiLU(), nn.Linear(dimension, 1))
        self.output = nn.Linear(dimension, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image_codes, history, grid, valid=None):
        if not history:
            return torch.zeros_like(image_codes)
        past = history[-1].detach()
        batch, patches, _ = image_codes.shape
        height, width = grid
        if past.shape != image_codes.shape or patches != height * width:
            raise ValueError("Historical image-code grids differ")
        position, integer = patch_positions(height, width, image_codes.device, image_codes.dtype)
        radius = self.window // 2
        dy, dx = torch.meshgrid(torch.arange(-radius, radius + 1, device=past.device),
                               torch.arange(-radius, radius + 1, device=past.device), indexing="ij")
        neighbors = integer[:, None] + torch.stack((dy.flatten(), dx.flatten()), -1)[None]
        available = ((neighbors[..., 0] >= 0) & (neighbors[..., 0] < height)
                     & (neighbors[..., 1] >= 0) & (neighbors[..., 1] < width))
        indices = (neighbors[..., 0].clamp(0, height - 1) * width
                   + neighbors[..., 1].clamp(0, width - 1)).long()
        query = self.query(image_codes)
        keys = self.key(past)[:, indices]
        values = self.value(past)[:, indices]
        relative = position[:, None] - position[indices]
        offset = torch.ones_like(relative[..., :1])
        bias = self.relative_bias(torch.cat((relative, offset), -1)).squeeze(-1)
        logits = (query[:, :, None].float() * keys.float()).sum(-1) / math.sqrt(self.dimension)
        logits = (logits + bias.float()[None]).masked_fill(~available[None], -torch.inf)
        attention = logits.softmax(-1).to(values.dtype)
        message = (attention[..., None] * values).sum(2)
        gate = self.gate(torch.cat((image_codes, message), -1)).sigmoid()
        residual = self.output(gate * message)
        if valid is not None:
            residual = residual * valid.to(residual.dtype).reshape(batch, 1, 1)
        return residual


class CoField(nn.Module):
    def __init__(self, classes=1, pretrained=True, temporal_dim=None, temporal_window=None):
        super().__init__()
        if classes < 1:
            raise ValueError("classes must be positive")
        self.classes = classes
        self.channels = 3 * classes
        self.image_encoder = timm.create_model("vit_small_patch16_224.augreg_in21k",
            pretrained=pretrained, num_classes=0, global_pool="", img_size=224, dynamic_img_size=True)
        self.feature_fusion = nn.Sequential(nn.LayerNorm(4 * 384), nn.Linear(4 * 384, 384))
        self.time_embedding = TimeEmbedding(384)
        self.context = nn.Parameter(torch.randn(1, 1, 384) * 0.02)
        self.weaver = FieldWeaver(self.channels)
        self.decoder = LocalFieldDecoder(self.channels)
        self.grid_head = nn.Sequential(RMSNorm(384), nn.Linear(384, self.channels))
        nn.init.zeros_(self.grid_head[1].weight)
        nn.init.zeros_(self.grid_head[1].bias)
        self.temporal = None
        if temporal_dim is not None:
            self.temporal = TemporalRouteAdapter(temporal_dim, temporal_window)
        self.gradient_checkpointing = False

    def encode_image(self, image):
        encoder = self.image_encoder
        tokens = encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(image))))
        features = []
        for index, block in enumerate(encoder.blocks):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in (2, 5, 8, 11):
                feature = encoder.norm(tokens) if index == 11 else tokens
                features.append(feature[:, encoder.num_prefix_tokens:])
        return self.feature_fusion(torch.cat(features, -1))

    def encode_condition(self, image, history=None, valid=None):
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("Image dimensions must be divisible by 16")
        codes = self.encode_image(image)
        residual = None
        if self.temporal is not None:
            grid = (image.shape[-2] // 16, image.shape[-1] // 16)
            residual = self.temporal(codes, history, grid, valid)
        return codes, residual

    def decode_state(self, condition, state, timestep):
        batch, channels, height, width = state.shape
        if channels != self.channels or height % 16 or width % 16:
            raise ValueError("Invalid state shape")
        time = torch.as_tensor(timestep, device=state.device, dtype=torch.float32)
        time = time.expand(batch) if time.ndim == 0 else time
        embedding = self.time_embedding(time)
        codes = self.weaver(condition[0], state, embedding, self.context, condition[1])
        codes = F.silu(codes + embedding[:, None] + self.context)
        grid = self.grid_head(codes).transpose(1, 2).reshape(batch, channels, height // 16, width // 16)
        return grid, self.decoder(codes, state)

    def forward(self, image, state, timestep, history=None, valid=None):
        return self.decode_state(self.encode_condition(image, history, valid), state, timestep)


class CoFieldV(CoField):
    def __init__(self, temporal_dim, temporal_window, pretrained=True):
        super().__init__(1, pretrained, temporal_dim, temporal_window)


def build_model(config, pretrained=False):
    if config["mode"] == "sunseg":
        return CoFieldV(int(config["temporal_dim"]), int(config["temporal_window"]), pretrained)
    if config["mode"] != "image":
        raise ValueError("mode must be image or sunseg")
    return CoField(config["classes"], pretrained)
