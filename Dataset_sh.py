import os

import numpy as np
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        default_feature_path = os.path.join(self.dataset_path, self.dataset_name, "features")
        self.feature_path = args.feature_path if getattr(args, "feature_path", "") else default_feature_path

        split_path = os.path.join(self.dataset_path, self.dataset_name, "test_split_10crop.txt")
        self.testlist = self._txt2list(split_path)

    def _txt2list(self, txtpath):
        with open(txtpath, mode="r") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]

    def __getitem__(self, index):
        data_video_name = self.testlist[index]
        feature_path = os.path.join(self.feature_path, data_video_name + ".npy")
        feature = np.load(feature_path).astype(np.float32)
        return feature, data_video_name

    def __len__(self):
        return len(self.testlist)
