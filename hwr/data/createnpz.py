import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from hwr.constants import SPLIT, PREPROCESS
from hwr.data.reader import IAMReader


# Pre-create appropriate preprocessed data and features as npz file format to
# save training time
# E.g. to create npz for preprocess scheme 6, do
# >> python createnpz.py 6


def save_npz(preprocess_no):
    preprocess = getattr(PREPROCESS, "SCHEME" + str(preprocess_no))
    reader = IAMReader(SPLIT.ALL)
    samples = reader.get_samples()
    bad_samples = []
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        xml_path = sample.xml_path
        y = sample.ground_truth
        f_split = xml_path.split('/')
        f_split[-4] = 'npz-' + preprocess_no
        f_split[-1] = f_split[-1][:-3] + 'npz'
        f = '/'.join(f_split)
        try:
            features = sample.generate_features(preprocess=preprocess)
            labels = y
            d = '/'.join(f_split[:-1])
            if not os.path.exists(d):
                os.makedirs(d)
            np.savez(f, x=features, y=labels)
        except ValueError:
            print("Bad sample: {}".format(f_split[-1]))
            bad_samples.append(f_split[-1])


if __name__ == "__main__":
    parser = ArgumentParser(description='Pre preprocess data into npz.')
    parser.add_argument("scheme",
                        help="number of preprocess scheme")
    args = parser.parse_args()
    save_npz(args.scheme)




