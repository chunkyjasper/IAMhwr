import random

import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from hwr.constants import SPLIT, PREPROCESS
from hwr.data.reader import IAMReader, xmlpath2npypath
from hwr.decoding.mlf import mlf2label, mlf2txt


# Generator of IAM-ON data for Keras model
class IAMSequence(Sequence):
    def __init__(self, split=SPLIT.ALL, batch_size=1, pred=False,
                 npz=False, preprocess=None, pad_to=None, inout_ratio=4):
        reader = IAMReader(split)
        self.samples = np.asarray(reader.get_samples())
        # pre-preprocessed data in npz
        self.npz = npz
        self.npz_dir = "npz-" + str(preprocess)
        # xs: features
        # ys: ground truths
        self.xs = []
        self.ys = []

        # Load features from npz or preprocess from scratch
        if not self.npz:
            print("Not using npz. Data preprocessing may take some time.")
            preprocess_scheme = getattr(PREPROCESS, "SCHEME" + str(preprocess))
            self.xs = np.asarray([s.generate_features(preprocess_scheme) for s in tqdm(self.samples)])
            self.ys = np.asarray([s.ground_truth for s in self.samples])

        else:
            for s in self.samples:
                data = np.load(xmlpath2npypath(s.xml_path, self.npz_dir))
                self.xs.append(data['x'])
                self.ys.append(data['y'])
            self.xs = np.asarray(self.xs)
            self.ys = np.asarray(self.ys)

        self.n = len(self.samples)
        # Indices for shuffling
        self.indices = np.arange(self.n)
        np.random.shuffle(self.indices)
        self.batch_size = batch_size
        # If pred, generate only xs
        self.pred = pred

        # Manually define pad value
        if pad_to:
            self.x_pad, self.y_pad = pad_to
            self.adaptive_pad = False

        # Else pad to match the longest sample in batch
        else:
            self.adaptive_pad = True
        # How much the TDNN scale down the input
        self.inout_ratio = inout_ratio


    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    # Get a batch
    def __getitem__(self, idx):
        # batch indices
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sample = self.samples[inds]
        batch_xs = self.xs[inds]
        batch_ys = self.ys[inds]

        # pad depending on the longest sample of each batch
        if self.adaptive_pad:
            max_len_x = max([len(i) for i in batch_xs])
            y_pad = int(np.ceil(max_len_x / self.inout_ratio))
            x_pad = y_pad * self.inout_ratio

        # Pad with given pad_x and pad_y value
        else:
            x_pad = self.x_pad
            y_pad = self.y_pad

        # features
        inputs = np.array([pad_2d(x, pad_to=x_pad, pad_value=0)
                           for x in batch_xs])
        # truth labels
        labels = np.array([pad_1d(y, pad_to=y_pad, pad_value=-1)
                           for y in mlf2label(batch_ys, multiple=True)])
        # Length of network output
        ypred_length = np.array([y_pad
                                 for _ in batch_sample])[:, np.newaxis]
        # Number of chars in ground truth
        ytrue_length = np.array([len(s.ground_truth)
                                 for s in batch_sample])[:, np.newaxis]
        # Prediction sequence, return only xs
        if self.pred:
            return inputs

        # Training/evaluation sequence
        return {'xs': inputs,
                'ys': labels,
                'ypred_length': ypred_length,
                'ytrue_length': ytrue_length}, labels

    # Get a random sample for demonstration/testing
    def random_sample(self, pad=10):
        idx = random.randint(0, self.n - 1)
        return self.sample_at_idx(idx, pad=pad)

    def sample_at_idx(self, idx, pad=10):
        idx = self.indices[idx]
        return self.sample_at_absolute_idx(idx, pad=pad)

    # regardless of shuffling
    def sample_at_absolute_idx(self, idx, pad=10):
        pointset = self.samples[idx].pointset
        ground_truth = mlf2txt(self.samples[idx].ground_truth)
        network_input = self.xs[idx]
        network_input = pad_2d(self.xs[idx],
                               pad_to=network_input.shape[0] + pad,
                               pad_value=0)
        network_input = np.asarray([network_input])
        return network_input, ground_truth, pointset


    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    # Get xs and ys which match the current permutation defined by indices
    def get_xy(self):
        xs = [self.xs[idx] for idx in self.indices]
        ys = mlf2txt([self.ys[idx] for idx in self.indices], multiple=True)
        return xs, ys

    def gen_iter(self):
        for i in range(len(self)):
            yield self[i]


def pad_2d(x, pad_to, pad_value):
    result = np.ones((pad_to, x.shape[1])) * pad_value
    result[:x.shape[0], :] = x
    return result


def pad_1d(x, pad_to, pad_value):
    result = np.ones(pad_to) * pad_value
    result[:x.shape[0]] = x
    return result
