import argparse


parser = argparse.ArgumentParser(description="UC_VAD test-only")

# runtime
parser.add_argument("--device", type=int, default=0, help="GPU ID (ignored when CUDA is unavailable)")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--eval_iter", type=int, default=0, help="iteration tag used in evaluation logs")

# io and checkpoint
parser.add_argument("--model_name", default="UC_VAD_testonly", help="evaluation run name")
parser.add_argument("--pretrained_ckpt", default="", help="path to pretrained model checkpoint (.pkl)")
parser.add_argument("--ckpt_root", default="./bestckpt", help="default ckpt root when --pretrained_ckpt is empty")
parser.add_argument("--plot", type=int, default=0, help="whether to plot anomaly maps")

# dataset
parser.add_argument("--dataset_name", type=str, default="ucf-crime", help="dataset name")
parser.add_argument("--dataset_path", type=str, default="./dataset", help="dataset root path")
parser.add_argument(
    "--feature_path",
    type=str,
    default="",
    help="path to extracted feature .npy files; defaults to <dataset_path>/<dataset_name>/features",
)
parser.add_argument("--feature_size", type=int, default=512, help="input feature dimension")

# dual-branch model
parser.add_argument("--proto_num", type=int, default=5, help="number of normal/abnormal prototypes")
parser.add_argument("--context_dropout", type=float, default=0.1, help="dropout applied to prototype contexts")
parser.add_argument("--head_dropout", type=float, default=0.1, help="dropout in branch classifiers")

# inference
parser.add_argument("--fusion_max_ratio", type=float, default=0.25, help="max-fusion ratio in adaptive branch fusion")
parser.add_argument(
    "--fusion_refined_scale",
    type=float,
    default=0.0,
    help="scale of refined-confidence preference in fusion",
)
parser.add_argument(
    "--fusion_refined_bias",
    type=float,
    default=0.0,
    help="bias of refined-confidence preference in fusion",
)
parser.add_argument(
    "--fusion_static_refined_weight",
    type=float,
    default=0.85,
    help="if >=0, use fixed refined weight instead of adaptive confidence fusion",
)
parser.add_argument("--test_smooth_kernel", type=int, default=13, help="smoothing kernel size for test scores")
parser.add_argument("--test_persist_alpha", type=float, default=0.95, help="temporal persistence decay (0 disables)")
parser.add_argument("--test_persist_blend", type=float, default=0.10, help="blend ratio of persistence enhancement")
parser.add_argument("--test_second_smooth_kernel", type=int, default=3, help="optional second smoothing kernel (0 disables)")
parser.add_argument("--test_score_gamma", type=float, default=0.85, help="power-law calibration on test scores")
parser.add_argument("--test_video_norm_mix", type=float, default=0.0, help="mix ratio of per-video score normalization")
parser.add_argument("--test_video_norm_temp", type=float, default=1.0, help="temperature for per-video normalization sigmoid")
parser.add_argument("--test_peak_boost", type=float, default=0.40, help="tail enhancement ratio for high-score peaks")
parser.add_argument("--test_peak_percentile", type=float, default=70.0, help="percentile threshold for peak enhancement")
parser.add_argument(
    "--test_fuse_after_avg",
    type=int,
    default=0,
    help="if 1, average branch scores across crops before branch fusion",
)
