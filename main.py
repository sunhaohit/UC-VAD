from __future__ import print_function

import datetime
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import options
from Dataset_sh import dataset as sh_dataset
from Dataset_ucf import dataset as ucf_dataset
from UC_VAD import UC_VAD
from eval import eval_p
from test import test


def _select_dataset(dataset_name):
    name = str(dataset_name).lower()
    if name == "ucf-crime":
        return ucf_dataset
    if name == "shanghaitech":
        return sh_dataset
    raise ValueError("Unsupported dataset_name: {} (expected 'ucf-crime' or 'shanghaitech')".format(dataset_name))


def _resolve_checkpoint_path(args):
    if args.pretrained_ckpt:
        ckpt_path = args.pretrained_ckpt
    else:
        ckpt_path = os.path.join(args.ckpt_root, "{}.pkl".format(args.dataset_name))

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Checkpoint not found: {}. Use --pretrained_ckpt or place default ckpt under --ckpt_root.".format(
                ckpt_path
            )
        )
    return ckpt_path


def _build_device(device_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    args = options.parser.parse_args()
    dataset_cls = _select_dataset(args.dataset_name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _build_device(args.device)
    time_now = datetime.datetime.now()
    save_path = os.path.join(
        args.model_name,
        "{}{:02d}{:02d}{:02d}{:02d}{:02d}".format(
            time_now.year, time_now.month, time_now.day, time_now.hour, time_now.minute, time_now.second
        ),
    )

    model = UC_VAD(feature_size=args.feature_size, args=args).to(device)
    ckpt_path = _resolve_checkpoint_path(args)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print("Model loaded weights from {}".format(ckpt_path))

    test_dataset = dataset_cls(args=args)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10,
        pin_memory=True,
        num_workers=1,
        shuffle=False,
    )

    model.eval()
    predict_dict = test(test_loader, model, device, args)
    auc = eval_p(
        itr=int(getattr(args, "eval_iter", 0)),
        dataset=args.dataset_name,
        predict_dict=predict_dict,
        logger=False,
        save_path=save_path,
        plot=args.plot,
        args=args,
    )
    print("Eval finished | auc={:.6f}".format(auc if auc is not None else float("nan")))
