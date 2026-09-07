import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import repeat
from .Attention_layers import AttentionLayer, SelfAttentionLayer, EmbeddedAttention


def maybe_cat_exog(x, u):
    if u is None:
        return x
    if u.dim() == 3:
        u = u.unsqueeze(2)
    expand_shape = list(u.shape)
    for dim, size in enumerate(x.shape[:-1]):
        if expand_shape[dim] == 1:
            expand_shape[dim] = size
    return torch.cat([x, u.expand(*expand_shape)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EmbeddedAttentionLayer(nn.Module):
    """
    Spatial embedded attention layer
    """
    def __init__(self,
                 model_dim, adaptive_embedding_dim, feed_forward_dim=2048, dropout=0):
        super().__init__()

        self.attn = EmbeddedAttention(model_dim, adaptive_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        # emb: (..., length, model_dim)
        residual = x
        out = self.attn(x, emb)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer
    """
    def __init__(self, seq_len, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super(ProjectedAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

        self.seq_len = seq_len

    def forward(self, x):
        # x: [b s n d]
        batch = x.shape[0]
        projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
                              repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]

        message_out = self.out_attn(projector, x, x)  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        message_in = self.in_attn(x, projector, message_out)  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class ImputeFormerModel(nn.Module):
    """
    Spatiotempoarl Imputation Transformer
    """
    def __init__(
            self,
            num_nodes,
            input_dim=6,
            output_dim=1,
            input_embedding_dim=24,
            learnable_embedding_dim=80,
            feed_forward_dim=256,
            num_temporal_heads=4,
            num_layers=3,
            dropout=0.,
            windows=24,
            dim_proj=10,
            variable_length=False,
            use_interpolation_residual=False,
            time_aware=False,
            static_sensor_embedding=False,
            visible_median_calibration=False,
            no_context_prior=None,
    ):
        super(ImputeFormerModel, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = windows
        self.out_steps = windows
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.learnable_embedding_dim = learnable_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + learnable_embedding_dim)
        self.num_temporal_heads = num_temporal_heads
        self.num_layers = num_layers
        self.variable_length = variable_length
        self.use_interpolation_residual = use_interpolation_residual
        self.time_aware = time_aware
        self.static_sensor_embedding = static_sensor_embedding
        self.visible_median_calibration = visible_median_calibration

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if use_interpolation_residual:
            self.mask_proj = nn.Linear(1, input_embedding_dim, bias=False)
        self.dim_proj = dim_proj

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(1 if static_sensor_embedding else windows, num_nodes, learnable_embedding_dim)))

        self.readout = MLP(self.model_dim, self.model_dim, output_dim, n_layers=2)
        if time_aware:
            self.residual_strength = nn.Parameter(torch.full((1, 1, num_nodes, 1), -2.0))
            self.register_buffer("context_residual_gain", torch.ones(num_nodes, 3))
            self.register_buffer("no_context_prior", None if no_context_prior is None else
                                 torch.as_tensor(no_context_prior, dtype=torch.float32).view(1, 1, num_nodes, 1))
            if visible_median_calibration:
                self.register_buffer("context_median_gain", torch.zeros(num_nodes, 3))
            self.local_mixer = nn.Conv2d(
                self.model_dim, self.model_dim, (5, 1), padding=(2, 0), groups=self.model_dim
            )

        self.attn_layers_t = nn.ModuleList(
            [ProjectedAttentionLayer(self.num_nodes, self.dim_proj, self.model_dim, num_temporal_heads, self.model_dim, dropout)
             for _ in range(num_layers)])

        self.attn_layers_s = nn.ModuleList(
            [EmbeddedAttentionLayer(self.model_dim, learnable_embedding_dim, feed_forward_dim)
                for _ in range(num_layers)])


    @staticmethod
    def visible_interpolation(x, mask):
        """Linear initial estimate using only values visible to this forward pass."""
        steps = x.shape[1]
        positions = torch.arange(steps, device=x.device).view(1, steps, 1, 1)
        visible = mask.bool()
        left = torch.where(visible, positions, -1).cummax(dim=1).values
        right = torch.where(visible, positions, steps).flip(1).cummin(dim=1).values.flip(1)
        left_value = x.gather(1, left.clamp(0, steps - 1))
        right_value = x.gather(1, right.clamp(0, steps - 1))
        fraction = (positions - left).to(x.dtype) / (right - left).clamp(min=1)
        interpolated = left_value + fraction * (right_value - left_value)
        interpolated = torch.where(left < 0, right_value, interpolated)
        interpolated = torch.where(right >= steps, left_value, interpolated)
        return torch.where((left < 0) & (right >= steps), 0.0, interpolated)

    @staticmethod
    def visible_median(x, mask):
        """Visible-only median, averaging the middle pair for even sample counts."""
        count = mask.bool().sum(1, keepdim=True)
        ordered = torch.where(mask.bool(), x, torch.inf).sort(dim=1).values
        lower = ordered.gather(1, ((count - 1) // 2).clamp(min=0))
        upper = ordered.gather(1, count // 2)
        return torch.where(count > 0, (lower + upper) / 2.0, 0.0)

    @staticmethod
    def time_aware_components(x, u, mask):
        """Derive physical-time context using only currently visible measurements."""
        steps = x.shape[1]
        visible = mask.bool()
        safe = torch.where(visible, x, 0.0)
        positions = torch.arange(steps, device=x.device).view(1, steps, 1, 1)
        left = torch.where(visible, positions, -1).cummax(1).values
        right = torch.where(visible, positions, steps).flip(1).cummin(1).values.flip(1)
        left_index, right_index = left.clamp(0, steps - 1), right.clamp(0, steps - 1)
        elapsed = (u[:, :, 0] - u[:, :1, 0])[:, :, None, None].expand_as(x)
        left_time, right_time = elapsed.gather(1, left_index), elapsed.gather(1, right_index)
        before, after = elapsed - left_time, right_time - elapsed
        fraction = (before / (right_time - left_time).clamp(min=1e-12)).to(x.dtype)
        left_value, right_value = safe.gather(1, left_index), safe.gather(1, right_index)
        baseline = left_value + fraction * (right_value - left_value)
        baseline = torch.where(left < 0, right_value, baseline)
        baseline = torch.where(right == steps, left_value, baseline)
        baseline = torch.where((left < 0) & (right == steps), 0.0, baseline)

        count = visible.sum(1, keepdim=True)
        location = safe.sum(1, keepdim=True) / count.clamp(min=1)
        variance = torch.where(visible, (safe - location).square(), 0.0).sum(1, keepdim=True) / count.clamp(min=1)
        scale = torch.where(count > 0, variance.clamp(min=0.05 ** 2).sqrt(), 1.0)
        span = elapsed[:, -1:, :, :].clamp(min=1e-12)

        def bounded_log(value):
            encoded = torch.log1p(value.clamp(min=0))
            return (encoded / (1.0 + encoded)).to(x.dtype)

        before_encoded = torch.where(left >= 0, bounded_log(before), 1.0)
        after_encoded = torch.where(right < steps, bounded_log(after), 1.0)
        delta = u[:, :, 1, None, None].expand_as(x).clone()
        if steps > 1:
            delta[:, 0] = elapsed[:, 1] - elapsed[:, 0]
        features = torch.cat([
            ((baseline - location) / scale).clamp(-8, 8),
            visible.to(x.dtype), bounded_log(delta), before_encoded, after_encoded,
            (left >= 0).to(x.dtype), (right < steps).to(x.dtype),
            bounded_log(span).expand_as(x), (elapsed / span).to(x.dtype),
            torch.tanh(baseline / 3.0), bounded_log(scale).expand_as(x),
            torch.tanh(location / 3.0).expand_as(x), bounded_log(count).expand_as(x),
        ], dim=-1)
        return baseline, features, scale

    def forward(self, x, u, mask):
        # x is the observation channel; u contains temporal covariates.
        batch_size = x.shape[0]
        if x.shape[1] < 1:
            raise ValueError("ImputeFormer needs at least one time point.")
        if self.time_aware:
            if self.visible_median_calibration:
                median = self.visible_median(x, mask)
                enough_visible = mask.bool().sum(1, keepdim=True) >= 3
            baseline, x, local_scale = self.time_aware_components(x, u, mask)
            context_class = (x[..., 5] + x[..., 6]).long()
            sensor_index = torch.arange(self.num_nodes, device=x.device).view(1, 1, -1)
            residual_gain = self.context_residual_gain[sensor_index, context_class][..., None]
        else:
            baseline = self.visible_interpolation(x, mask) if self.use_interpolation_residual else None
            x = baseline if baseline is not None else x * mask
            x = maybe_cat_exog(x, u)
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        if self.use_interpolation_residual:
            x = x + self.mask_proj(mask.to(x.dtype))

        embedding = self.learnable_embedding
        if self.static_sensor_embedding:
            embedding = embedding.expand(x.shape[1], -1, -1)
        elif x.shape[1] != embedding.shape[0]:
            if not self.variable_length:
                raise ValueError("This legacy checkpoint requires its configured window length.")
            embedding = F.interpolate(
                embedding.permute(1, 2, 0), size=x.shape[1], mode="linear", align_corners=False
            ).permute(2, 0, 1)
        node_emb = embedding.expand(batch_size, *embedding.shape)
        x = torch.cat([x, node_emb], dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        if self.time_aware:
            x = x + self.local_mixer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = x.permute(0, 2, 1, 3)  # [b n s c]
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, embedding, dim=1)

        x = x.permute(0, 2, 1, 3)  # [b s n c]
        out = self.readout(x)
        if self.time_aware:
            out = baseline + residual_gain * 2.0 * local_scale * self.residual_strength.sigmoid() * torch.tanh(out)
            if self.visible_median_calibration:
                median_gain = self.context_median_gain[sensor_index, context_class][..., None]
                out = out + torch.where(enough_visible, median_gain * (median - baseline), 0.0)
            if self.no_context_prior is not None:
                out = out + torch.where(context_class[..., None] == 0, self.no_context_prior, 0.0)
        elif baseline is not None:
            out = baseline + out

        return out


    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--input-dim', type=int, default=6)
        parser.add_argument('--num-nodes', type=int, default=207)
        parser.add_argument('--output-dim', type=int, default=1)
        parser.add_argument('--input-embedding-dim', type=int, default=24)
        parser.add_argument('--feed-forward-dim', type=int, default=256)
        parser.add_argument('--learnable-embedding-dim', type=int, default=80)
        parser.add_argument('--num_temporal_heads', type=int, default=4)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--dim-proj', type=int, default=10)
        parser.add_argument('--dropout', type=int, default=0.1)
        return parser
