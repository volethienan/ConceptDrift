from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class AdaptationBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnlineMLP(nn.Module):
    """MLP backbone exposing representations and layer-wise parameter groups."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        dims = [input_dim, *hidden_dims]
        self.blocks = nn.ModuleList(
            AdaptationBlock(dims[i], dims[i + 1], dropout) for i in range(len(hidden_dims))
        )
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)

    @property
    def layer_names(self) -> list[str]:
        return [f"layer_{i + 1}" for i in range(len(self.blocks))]

    def forward(
        self, x: torch.Tensor, return_representations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        representations: dict[str, torch.Tensor] = {}
        for name, block in zip(self.layer_names, self.blocks):
            x = block(x)
            representations[name] = x
        logits = self.classifier(x)
        if return_representations:
            return logits, representations
        return logits

    def adaptation_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {
            name: list(block.parameters())
            for name, block in zip(self.layer_names, self.blocks)
        }
        # The proposal allocates plasticity to representation layers.  Keeping
        # the task head separate avoids accidentally freezing label adaptation
        # just because the last hidden representation is protected.
        groups["classifier"] = list(self.classifier.parameters())
        return groups


class FusedAdapterInferenceMLP(OnlineMLP):
    """Deployment-only MLP with adapter deltas folded into linear weights.

    The training model remains structurally frozen and keeps its adapters for
    future online updates. This cache is rebuilt after a commit and is never
    optimized directly.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: list[int],
        dropout: float,
        input_adapter: nn.Module,
        input_adapter_enabled: bool,
    ):
        super().__init__(input_dim, n_classes, hidden_dims, dropout)
        self.input_adapter = copy.deepcopy(input_adapter)
        self.input_adapter_enabled = bool(input_adapter_enabled)

    def forward(
        self, x: torch.Tensor, return_representations: bool = False
    ) -> torch.Tensor:
        if return_representations:
            raise RuntimeError(
                "FusedAdapterInferenceMLP is prediction-only; use the "
                "unfused training model for drift representations"
            )
        original = x
        logits = super().forward(x)
        if self.input_adapter_enabled:
            logits = logits + self.input_adapter(original)
        return logits


class LowRankResidualAdapter(nn.Module):
    """Zero-initialized low-rank residual branch for a frozen linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int = 8,
        scale: float = 1.0,
    ):
        super().__init__()
        if max_rank < 1:
            raise ValueError("max_rank must be positive")
        self.max_rank = int(max_rank)
        self.active_rank = 0
        self.trainable_rank = 0
        self.scale = float(scale)
        self.down = nn.Parameter(torch.empty(self.max_rank, in_features))
        self.up = nn.Parameter(torch.zeros(out_features, self.max_rank))
        nn.init.kaiming_uniform_(self.down, a=5**0.5)

    def set_active_rank(self, rank: int, trainable_rank: int | None = None) -> None:
        self.active_rank = int(max(0, min(int(rank), self.max_rank)))
        requested_trainable = self.active_rank if trainable_rank is None else trainable_rank
        self.trainable_rank = int(
            max(0, min(int(requested_trainable), self.active_rank))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank = self.active_rank
        if rank == 0:
            return x.new_zeros((*x.shape[:-1], self.up.shape[0]))
        trainable = self.trainable_rank
        down = self.down[:rank]
        up = self.up[:, :rank]
        if trainable < rank:
            down = torch.cat([down[:trainable], down[trainable:].detach()], dim=0)
            up = torch.cat([up[:, :trainable], up[:, trainable:].detach()], dim=1)
        residual = F.linear(F.linear(x, down), up)
        return residual * self.scale


class FeatureWiseResidualModulator(nn.Module):
    """Zero-initialized bounded affine modulation of a hidden representation.

    Unlike the low-rank branch, this adapter acts after normalization and the
    non-linearity. It can therefore recalibrate every hidden feature without
    changing any frozen backbone parameter.
    """

    def __init__(self, features: int, scale: float = 0.5):
        super().__init__()
        self.scale = float(scale)
        self.gain = nn.Parameter(torch.zeros(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = 1.0 + self.scale * torch.tanh(self.gain)
        bias = self.scale * torch.tanh(self.bias)
        return x * gain + bias


class AdapterOnlineMLP(OnlineMLP):
    """Sequential MLP with trainable low-rank branches at every hidden layer.

    The base blocks are trained normally during warm-up.  Afterwards
    ``freeze_backbone`` makes the no-global-update invariant structural:
    adaptation groups expose only adapter parameters and the classifier.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.0,
        adapter_max_rank: int = 8,
        adapter_scale: float = 1.0,
        input_adapter_dim: int = 16,
        modulation_scale: float = 0.5,
    ):
        super().__init__(input_dim, n_classes, hidden_dims, dropout)
        dims = [input_dim, *hidden_dims]
        self.adapters = nn.ModuleList(
            LowRankResidualAdapter(
                dims[i],
                dims[i + 1],
                max_rank=adapter_max_rank,
                scale=adapter_scale,
            )
            for i in range(len(hidden_dims))
        )
        if input_adapter_dim < 1:
            raise ValueError("input_adapter_dim must be positive")
        self.input_adapter = nn.Sequential(
            nn.Linear(input_dim, input_adapter_dim),
            nn.ReLU(),
            nn.Linear(input_adapter_dim, n_classes, bias=False),
        )
        self.modulators = nn.ModuleList(
            FeatureWiseResidualModulator(width, modulation_scale)
            for width in hidden_dims
        )
        nn.init.zeros_(self.input_adapter[-1].weight)
        self.input_adapter_enabled = False
        self.feature_modulation_enabled = False

    @property
    def adapter_rank(self) -> int:
        return max((adapter.active_rank for adapter in self.adapters), default=0)

    @property
    def adapter_trainable_rank(self) -> int:
        return max((adapter.trainable_rank for adapter in self.adapters), default=0)

    def set_adapter_rank(
        self, rank: int, trainable_rank: int | None = None
    ) -> None:
        for adapter in self.adapters:
            adapter.set_active_rank(rank, trainable_rank)

    def freeze_backbone(self) -> None:
        for parameter in self.blocks.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapters.parameters():
            parameter.requires_grad_(True)
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(True)
        for parameter in self.input_adapter.parameters():
            parameter.requires_grad_(True)
        for parameter in self.modulators.parameters():
            parameter.requires_grad_(self.feature_modulation_enabled)

    def enable_input_adapter(self, enabled: bool = True) -> None:
        self.input_adapter_enabled = bool(enabled)

    def enable_feature_modulation(self, enabled: bool = True) -> None:
        self.feature_modulation_enabled = bool(enabled)

    def forward(
        self, x: torch.Tensor, return_representations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        original = x
        representations: dict[str, torch.Tensor] = {}
        for name, block, adapter, modulator in zip(
            self.layer_names, self.blocks, self.adapters, self.modulators
        ):
            x = block.net[0](x) + adapter(x)
            for operation in block.net[1:]:
                x = operation(x)
            if self.feature_modulation_enabled:
                x = modulator(x)
            representations[name] = x
        logits = self.classifier(x)
        if self.input_adapter_enabled:
            logits = logits + self.input_adapter(original)
        if return_representations:
            return logits, representations
        return logits

    def adaptation_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            name: list(adapter.parameters())
            for name, adapter in zip(self.layer_names, self.adapters)
        }
        groups["classifier"] = list(self.classifier.parameters())
        groups["input_adapter"] = list(self.input_adapter.parameters())
        if self.feature_modulation_enabled:
            groups.update(
                {
                    f"modulation_{name}": list(modulator.parameters())
                    for name, modulator in zip(
                        self.layer_names, self.modulators
                    )
                }
            )
        return groups

    def to_fused_inference(self) -> FusedAdapterInferenceMLP:
        """Return an inference cache algebraically equivalent to this model."""

        hidden_dims = [
            int(block.net[0].out_features) for block in self.blocks
        ]
        first_linear = self.blocks[0].net[0]
        dropout = next(
            (
                float(module.p)
                for module in self.blocks[0].net
                if isinstance(module, nn.Dropout)
            ),
            0.0,
        )
        fused = FusedAdapterInferenceMLP(
            int(first_linear.in_features),
            int(self.classifier.out_features),
            hidden_dims,
            dropout,
            self.input_adapter,
            self.input_adapter_enabled,
        ).to(first_linear.weight.device)

        previous_gain: torch.Tensor | None = None
        previous_bias: torch.Tensor | None = None
        with torch.no_grad():
            for index, (source_block, adapter, modulator) in enumerate(
                zip(self.blocks, self.adapters, self.modulators)
            ):
                fused.blocks[index] = copy.deepcopy(source_block)
                source_linear = source_block.net[0]
                weight = source_linear.weight.detach().clone()
                rank = int(adapter.active_rank)
                if rank > 0:
                    weight = weight + adapter.scale * (
                        adapter.up[:, :rank] @ adapter.down[:rank]
                    )
                bias = source_linear.bias.detach().clone()
                if previous_gain is not None and previous_bias is not None:
                    bias = bias + weight @ previous_bias
                    weight = weight * previous_gain.unsqueeze(0)
                fused_linear = fused.blocks[index].net[0]
                fused_linear.weight.copy_(weight)
                fused_linear.bias.copy_(bias)

                if self.feature_modulation_enabled:
                    previous_gain = (
                        1.0
                        + modulator.scale
                        * torch.tanh(modulator.gain.detach())
                    )
                    previous_bias = (
                        modulator.scale
                        * torch.tanh(modulator.bias.detach())
                    )
                else:
                    previous_gain = None
                    previous_bias = None

            classifier_weight = self.classifier.weight.detach().clone()
            classifier_bias = self.classifier.bias.detach().clone()
            if previous_gain is not None and previous_bias is not None:
                classifier_bias = (
                    classifier_bias + classifier_weight @ previous_bias
                )
                classifier_weight = (
                    classifier_weight * previous_gain.unsqueeze(0)
                )
            fused.classifier.weight.copy_(classifier_weight)
            fused.classifier.bias.copy_(classifier_bias)
            fused.input_adapter.load_state_dict(
                copy.deepcopy(self.input_adapter.state_dict())
            )
        fused.requires_grad_(False)
        return fused.eval()


class ModularOnlineMLP(nn.Module):
    """Parallel feature-group MLP with independently adaptable modules.

    Unlike depth-wise blocks, a shift confined to one feature group does not
    mechanically propagate through every adaptation module.  This makes the
    selective-plasticity hypothesis identifiable on controlled streams.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.0,
        n_modules: int = 4,
        module_dim: int = 64,
        head_dim: int | None = None,
        feature_groups: list[list[int]] | None = None,
    ):
        super().__init__()
        if module_dim < 1:
            raise ValueError("module_dim must be positive")
        if feature_groups is None:
            if n_modules < 2 or n_modules > input_dim:
                raise ValueError("n_modules must be between 2 and input_dim")
            boundaries = torch.linspace(0, input_dim, n_modules + 1).round().int().tolist()
            self.feature_slices = [
                (boundaries[i], boundaries[i + 1]) for i in range(n_modules)
            ]
            self.feature_groups = None
            widths = [stop - start for start, stop in self.feature_slices]
        else:
            flat = [index for group in feature_groups for index in group]
            if sorted(flat) != list(range(input_dim)) or any(not group for group in feature_groups):
                raise ValueError("feature_groups must partition every input column")
            n_modules = len(feature_groups)
            self.feature_slices = []
            self.feature_groups = [list(group) for group in feature_groups]
            for index, group in enumerate(self.feature_groups):
                self.register_buffer(f"feature_index_{index}", torch.tensor(group, dtype=torch.long))
            widths = [len(group) for group in self.feature_groups]
        self.modules_list = nn.ModuleList(
            AdaptationBlock(width, module_dim, dropout) for width in widths
        )
        if head_dim is not None and head_dim < 0:
            raise ValueError("head_dim must be None, 0 (linear) or positive")
        if head_dim == 0:
            # A linear head keeps most parameters inside the modules, so
            # module-level locality also bounds most of the parameter change.
            head: list[nn.Module] = [nn.Linear(n_modules * module_dim, n_classes)]
        else:
            fusion_dim = (
                int(head_dim)
                if head_dim is not None
                else int(hidden_dims[-1]) if hidden_dims else module_dim
            )
            head = [
                nn.Linear(n_modules * module_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
            ]
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            head.append(nn.Linear(fusion_dim, n_classes))
        self.classifier = nn.Sequential(*head)
        self.classifier.out_features = n_classes

    @property
    def layer_names(self) -> list[str]:
        return [f"module_{i + 1}" for i in range(len(self.modules_list))]

    def forward(
        self, x: torch.Tensor, return_representations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        representations: dict[str, torch.Tensor] = {}
        values: list[torch.Tensor] = []
        for index, (name, module) in enumerate(zip(self.layer_names, self.modules_list)):
            if self.feature_groups is None:
                start, stop = self.feature_slices[index]
                inputs = x[:, start:stop]
            else:
                inputs = x.index_select(1, getattr(self, f"feature_index_{index}"))
            value = module(inputs)
            representations[name] = value
            values.append(value)
        logits = self.classifier(torch.cat(values, dim=1))
        if return_representations:
            return logits, representations
        return logits

    def adaptation_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            name: list(module.parameters())
            for name, module in zip(self.layer_names, self.modules_list)
        }
        groups["classifier"] = list(self.classifier.parameters())
        return groups


class CDDAAttentionNet(nn.Module):
    """LSTM and multi-head self-attention backbone used by CDDA-MD.

    A tabular flow is treated as an ordered feature sequence.  This preserves
    the paper's recurrent/attention mechanism while accepting the exact same
    preprocessed tensors as every other method in the benchmark.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 32,
        attention_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("cdda_hidden_dim must be divisible by cdda_attention_heads")
        self.input_dim = input_dim
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @property
    def layer_names(self) -> list[str]:
        return ["lstm", "attention"]

    def forward(
        self, x: torch.Tensor, return_representations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sequence = x.unsqueeze(-1)
        recurrent, _ = self.lstm(sequence)
        attended, _ = self.attention(recurrent, recurrent, recurrent, need_weights=False)
        pooled = attended.mean(dim=1)
        logits = self.classifier(pooled)
        if return_representations:
            return logits, {
                "lstm": recurrent.mean(dim=1),
                "attention": pooled,
            }
        return logits

    def adaptation_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "lstm": list(self.lstm.parameters()),
            "attention": list(self.attention.parameters()),
            "classifier": list(self.classifier.parameters()),
        }
