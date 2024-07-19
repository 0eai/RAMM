import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
import numpy as np


class MuSeDataset(Dataset):
    def __init__(self, data, partition):
        super(MuSeDataset, self).__init__()
        self.partition = partition
        features_list, labels = data[partition]['features'], data[partition]['label']
        metas = data[partition]['meta']
        self.feature_dims = [features.shape[-1] for features in features_list[0]]
        self.n_samples = len(features_list)

        feature_lens = []
        label_lens = []
        for features in features_list:
            lens = [len(feature) for feature in features]
            feature_lens.append(lens)
        label_lens.append(1)
        
        # for lens in zip(*feature_lens):
        #     print(lens)
        
        max_feature_lens = [max(lens) for lens in zip(*feature_lens)]
        max_label_len = np.max(np.array(label_lens))
        
        self.feature_lens = torch.tensor(feature_lens)
        
        # print('self.feature_lens:', self.feature_lens)
        
        padded_features_list = []
        for features in features_list:
            padded_features = [
                np.pad(f, pad_width=((0, max_len - f.shape[0]), (0, 0))) 
                for f, max_len in zip(features, max_feature_lens)
            ]
            padded_features_list.append(padded_features)
        
        self.features_list = [
            [torch.tensor(f, dtype=torch.float).T for f in features] 
            for features in padded_features_list
        ]

        if max_label_len > 1:
            labels = [np.pad(l, pad_width=((0, max_label_len - l.shape[0]), (0, 0))) for l in labels]
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)
        self.metas = metas
        pass

    def get_feature_dim(self):
        return self.feature_dims

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        :param idx:
        :return: features, feature_lens, label, meta
            features: list of tensors with shape (seq_len, feature_dim)
            feature_lens: tensor of int tensors, lengths of the feature tensors before padding
            label: tensor of corresponding label(s) (shape 1 for n-to-1, else (seq_len, 1))
            meta: list containing corresponding meta data
        """
        features = self.features_list[idx]
        feature_lens = self.feature_lens[idx]
        label = self.labels[idx]
        meta = self.metas[idx]

        sample = features, feature_lens, label, meta
        return sample

def custom_collate_fn(data):
    """
    Custom collate function to ensure that the meta data are not treated with standard collate, but kept as ndarrays
    :param data:
    :return:
    """
    tensors = [d[:3] for d in data]
    np_arrs = [d[3] for d in data]
    coll_features, coll_feature_lens, coll_labels = default_collate(tensors)
    np_arrs_coll = np.row_stack(np_arrs)
    return coll_features, coll_feature_lens, coll_labels, np_arrs_coll