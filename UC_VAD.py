import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0)


class ProgressiveRefinementLayer(nn.Module):
    def __init__(self, feature_dim=512, num_prototypes=5, context_dropout=0.1):
        super().__init__()
        self.normal_proto_keys = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.normal_proto_values = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.abnormal_proto_keys = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.abnormal_proto_values = nn.Parameter(torch.randn(num_prototypes, feature_dim))

        self.scale = feature_dim ** -0.5
        self.context_norm = nn.LayerNorm(feature_dim)
        self.context_dropout = nn.Dropout(context_dropout)

        torch_init.normal_(self.normal_proto_keys, std=0.02)
        torch_init.normal_(self.normal_proto_values, std=0.02)
        torch_init.normal_(self.abnormal_proto_keys, std=0.02)
        torch_init.normal_(self.abnormal_proto_values, std=0.02)

    def forward(self, features):
        attn_n_logits = torch.matmul(features, self.normal_proto_keys.t()) * self.scale
        attn_n_weights = F.softmax(attn_n_logits, dim=-1)
        normal_context = torch.matmul(attn_n_weights, self.normal_proto_values)

        attn_a_logits = torch.matmul(features, self.abnormal_proto_keys.t()) * self.scale
        attn_a_weights = F.softmax(attn_a_logits, dim=-1)
        abnormal_context = torch.matmul(attn_a_weights, self.abnormal_proto_values)

        normal_context = self.context_dropout(self.context_norm(normal_context))
        abnormal_context = self.context_dropout(self.context_norm(abnormal_context))
        return normal_context, abnormal_context


class BranchClassifier(nn.Module):
    def __init__(self, feature_size=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class UC_VAD(nn.Module):
    def __init__(self, feature_size=512, args=None):
        super().__init__()
        num_prototypes = getattr(args, "proto_num", 5)
        context_dropout = getattr(args, "context_dropout", 0.1)
        head_dropout = getattr(args, "head_dropout", 0.2)

        self.prototype_layer = ProgressiveRefinementLayer(
            feature_dim=feature_size,
            num_prototypes=num_prototypes,
            context_dropout=context_dropout,
        )

        self.baseline_norm = nn.LayerNorm(feature_size)
        self.refined_norm = nn.LayerNorm(feature_size)
        self.baseline_classifier = BranchClassifier(feature_size=feature_size, dropout=head_dropout)
        self.refined_classifier = BranchClassifier(feature_size=feature_size, dropout=head_dropout)

        self.base_scale = nn.Parameter(torch.tensor(10.0))
        self.normal_scale = nn.Parameter(torch.tensor(5.0))
        self.abnormal_scale = nn.Parameter(torch.tensor(9.0))
        self.gate_scale = nn.Parameter(torch.tensor(8.0))
        self.gate_blend_logit = nn.Parameter(torch.tensor(2.2))

        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def _to_length_tensor(self, seq_len, x):
        if seq_len is None:
            return torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        if not torch.is_tensor(seq_len):
            seq_len = torch.as_tensor(seq_len, dtype=torch.long, device=x.device)
        else:
            seq_len = seq_len.to(device=x.device, dtype=torch.long)
        return torch.clamp(seq_len, min=1, max=x.size(1))

    def forward(self, x, seq_len=None):
        seq_len = self._to_length_tensor(seq_len, x)
        normal_context, abnormal_context = self.prototype_layer(x)

        base_scale = torch.clamp(self.base_scale, min=1.0, max=20.0)
        normal_scale = torch.clamp(self.normal_scale, min=0.1, max=20.0)
        abnormal_scale = torch.clamp(self.abnormal_scale, min=0.1, max=20.0)

        feat_baseline = self.baseline_norm(base_scale * x + normal_scale * normal_context)
        logits_baseline = self.baseline_classifier(feat_baseline)
        scores_baseline = self.sigmoid(logits_baseline)

        sim_n = F.cosine_similarity(x, normal_context, dim=-1).unsqueeze(-1)
        sim_a = F.cosine_similarity(x, abnormal_context, dim=-1).unsqueeze(-1)
        proto_gate = torch.sigmoid((sim_a - sim_n) * torch.clamp(self.gate_scale, min=1.0, max=20.0))
        score_gate = scores_baseline.detach()

        blend = torch.sigmoid(self.gate_blend_logit)
        # Keep dual-branch interaction but bias the gate toward prototype evidence
        gate = blend * proto_gate + (1.0 - blend) * score_gate
        gate = 0.85 * gate + 0.15 * proto_gate

        uncertainty = 1.0 - torch.abs(2.0 * score_gate - 1.0)
        gate = gate * (1.0 - 0.15 * uncertainty)
        gate = torch.clamp(gate, min=0.0, max=1.0)

        feat_refined = self.refined_norm(
            base_scale * x + normal_scale * normal_context + abnormal_scale * gate * abnormal_context
        )
        logits_refined = self.refined_classifier(feat_refined)
        scores_refined = self.sigmoid(logits_refined)
        return scores_baseline, scores_refined
