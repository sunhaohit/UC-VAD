import numpy as np
import torch


def _moving_average(scores, kernel_size):
    if kernel_size <= 1 or len(scores) <= kernel_size:
        return scores
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(scores, kernel, mode="same")


def _adaptive_branch_fusion(scores_b, scores_r, max_ratio=0.35, refined_scale=8.0, refined_bias=0.8):
    confidence_b = torch.mean(torch.abs(scores_b - 0.5), dim=1, keepdim=True)
    confidence_r = torch.mean(torch.abs(scores_r - 0.5), dim=1, keepdim=True)

    weight_r = torch.sigmoid(refined_scale * (confidence_r - confidence_b) + refined_bias)
    weight_r = 0.15 + 0.75 * weight_r
    weight_b = 1.0 - weight_r

    soft_fused = weight_b * scores_b + weight_r * scores_r
    hard_fused = torch.max(scores_b, scores_r)
    return max_ratio * hard_fused + (1.0 - max_ratio) * soft_fused


def _static_branch_fusion(scores_b, scores_r, refined_weight=0.5, max_ratio=0.35):
    refined_weight = float(np.clip(refined_weight, 0.0, 1.0))
    soft_fused = (1.0 - refined_weight) * scores_b + refined_weight * scores_r
    hard_fused = torch.max(scores_b, scores_r)
    return max_ratio * hard_fused + (1.0 - max_ratio) * soft_fused


def _persistence_enhance(scores, alpha, blend):
    if alpha <= 0.0 or blend <= 0.0:
        return scores
    alpha = float(np.clip(alpha, 0.0, 0.9999))
    blend = float(np.clip(blend, 0.0, 1.0))

    forward = np.zeros_like(scores, dtype=np.float32)
    backward = np.zeros_like(scores, dtype=np.float32)
    forward[0] = scores[0]
    for t in range(1, len(scores)):
        forward[t] = max(scores[t], alpha * forward[t - 1])
    backward[-1] = scores[-1]
    for t in range(len(scores) - 2, -1, -1):
        backward[t] = max(scores[t], alpha * backward[t + 1])
    persistent = np.maximum(forward, backward)
    return (1.0 - blend) * scores + blend * persistent


def _postprocess_scores(scores, gamma, norm_mix, norm_temp, peak_boost, peak_percentile):
    x = np.asarray(scores, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)

    if norm_mix > 0.0:
        mean = float(np.mean(x))
        std = float(np.std(x) + 1e-6)
        z = (x - mean) / std
        zsig = 1.0 / (1.0 + np.exp(-float(norm_temp) * z))
        x = (1.0 - float(norm_mix)) * x + float(norm_mix) * zsig

    if gamma != 1.0:
        x = np.power(np.clip(x, 1e-6, 1.0), float(gamma))

    if peak_boost > 0.0:
        threshold = float(np.percentile(x, float(peak_percentile)))
        x = x + float(peak_boost) * np.maximum(x - threshold, 0.0)

    return np.clip(x, 0.0, 1.0)


def test(test_loader, model, device, args):
    result = {}
    fuse_after_avg = int(getattr(args, "test_fuse_after_avg", 0)) > 0
    for _, data in enumerate(test_loader):
        feature, data_video_name = data
        feature = feature.to(device)
        seq_len = torch.sum(torch.max(torch.abs(feature), dim=2)[0] > 0, dim=1).long()

        with torch.no_grad():
            scores_b, scores_r = model(feature, seq_len)
            if fuse_after_avg:
                # Sweep-compatible path: average branch outputs across crops first, then fuse branches.
                scores_b = torch.mean(scores_b, dim=0, keepdim=True)
                scores_r = torch.mean(scores_r, dim=0, keepdim=True)
            if getattr(args, "fusion_static_refined_weight", -1.0) >= 0.0:
                fused_scores = _static_branch_fusion(
                    scores_b=scores_b,
                    scores_r=scores_r,
                    refined_weight=args.fusion_static_refined_weight,
                    max_ratio=args.fusion_max_ratio,
                )
            else:
                fused_scores = _adaptive_branch_fusion(
                    scores_b=scores_b,
                    scores_r=scores_r,
                    max_ratio=args.fusion_max_ratio,
                    refined_scale=args.fusion_refined_scale,
                    refined_bias=args.fusion_refined_bias,
                )
            fused_scores = torch.mean(fused_scores, dim=0).cpu().numpy().flatten()

        fused_scores = _moving_average(fused_scores, kernel_size=args.test_smooth_kernel)
        fused_scores = _persistence_enhance(
            fused_scores,
            alpha=getattr(args, "test_persist_alpha", 0.0),
            blend=getattr(args, "test_persist_blend", 0.0),
        )
        second_kernel = int(getattr(args, "test_second_smooth_kernel", 0))
        if second_kernel > 1:
            fused_scores = _moving_average(fused_scores, kernel_size=second_kernel)
        fused_scores = _postprocess_scores(
            fused_scores,
            gamma=getattr(args, "test_score_gamma", 1.0),
            norm_mix=getattr(args, "test_video_norm_mix", 0.0),
            norm_temp=getattr(args, "test_video_norm_temp", 1.0),
            peak_boost=getattr(args, "test_peak_boost", 0.0),
            peak_percentile=getattr(args, "test_peak_percentile", 90.0),
        )
        result[data_video_name[0]] = fused_scores
    return result
