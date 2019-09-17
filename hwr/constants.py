from enum import Enum
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"


class ON:
    class PATH:
        DATA_DIR = BASE_DIR + "../data/iamon/"
        MODEL_DIR = BASE_DIR + "../models/iamon/"
        SPLIT_CONFIG_DIR = DATA_DIR + "split-config/"
        LINE_DATA_DIR = DATA_DIR + "lines/"
        CKPT_DIR = MODEL_DIR + "checkpoint/"

    class PREPROCESS:
        # def preprocess(self, down_d=0, down_cos=1, slop_correction=False,
        #                normalize=False, resample_distance=0, up_sample=0)

        SCHEME1 = {'slope_correction': True,
                   'normalize': True,
                   'resample_distance': 0.22,
                   'up_sample': 10}
        #
        SCHEME2 = {
            'normalize': True,
            'down_d': 0.4}

        SCHEME3 = {
            'slope_correction': True,
            'normalize': True,
            'down_d': 0.4}

        SCHEME4 = {'slope_correction': True,
                   'normalize': True,
                   'down_cos': 0.975,
                   'resample_distance': 0.37,
                   'up_sample': 6}

        SCHEME5 = {'slope_correction': True,
                   'normalize': True,
                   'down_cos': 0.985,
                   'resample_distance': 0.26,
                   'up_sample': 10}

        SCHEME6 = {'slope_correction': True,
                   'normalize': True,
                   'down_cos': 0.975,
                   'resample_distance': 0.37,
                   'up_sample': 9}

        SCHEME7 = {'slope_correction': True,
                   'normalize': True,
                   'down_cos': 0.975,
                   'resample_distance': 0.30,
                   'up_sample': 12}

        CURRENT_SCHEME = SCHEME6


    class DATA:
        BLACKLIST = ["h02-037-02",
                     "a02-062-01",
                     "a02-062-02",
                     "a02-062-03",
                     "e05-265z-04",
                     'g04-060-02',
                     'j06-278z-04',
                     'h04-141z-03']

        CHARS_MLF = ['ex', 'qu', 'ga', 'do', 'am', 'ti', 'sl', 'lb', 'rb', 'ls', 'rs', 'sr', 'cm', 'mi',
                     'pl', 'pt', 'sp', 'cl', 'sc', 'qm', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
                     'n8', 'n9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                     'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                     'y', 'z']

        MLF_MAPPING = {"ex": "!",
                       "qu": '"',
                       "ga": "",  # crossing
                       "do": "",  # ?????
                       "am": "&",
                       "ti": "'",
                       "sl": "/",
                       "lb": "(",
                       "rb": ")",
                       "ls": "[",
                       "rs": "]",
                       "sr": "*",
                       "cm": ",",
                       "mi": "-",
                       "pl": "+",
                       "pt": ".",
                       "sp": " ",
                       "cl": ":",
                       "sc": ";",
                       "qm": "?",
                       "n0": "0",
                       "n1": "1",
                       "n2": "2",
                       "n3": "3",
                       "n4": "4",
                       "n5": "5",
                       "n6": "6",
                       "n7": "7",
                       "n8": "8",
                       "n9": "9", }

        CHARS = ['!', '"', '', '', '&', "'", '/', '(', ')', '[', ']', '*', ',', '-',
                 '+', '.', ' ', ':', ';', '?', '0', '1', '2', '3', '4', '5', '6', '7',
                 '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                 'y', 'z', '%']

        NON_ALPHAS = ['!', '"', '', '', '&', "'", '/', '(', ')', '[', ']', '*', ',', '-',
                      '+', '.', ' ', ':', ';', '?', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', ]

        CHARS_SIZE = len(CHARS_MLF) + 1  # +1 for blank for CTC
        BLANK_IDX = len(CHARS_MLF)


class DECODER(Enum):
    TRIE_BEAM_SEARCH = 1
    BEST_PATH = 2
    VANILLA_BEAM_SEARCH = 3


class SPLIT:
    TRAIN = "trainset.txt"
    TEST = "testset_f.txt"
    VAL1 = "testset_v.txt"
    VAL2 = "testset_t.txt"
    TEST_EXAMPLE = "test_example.txt"
    ALL = "all"


PRETRAINED = {
    "ONNET1": ON.PATH.MODEL_DIR + "ONNET1/2019-02-27-16:43:25/weights.h5",
    "ONNET2": ON.PATH.MODEL_DIR + "ONNET2/2019-02-28-02:06:14/weights.h5",
    "ONNET3": ON.PATH.MODEL_DIR + "ONNET3/2019-02-28-19:13:08/weights.h5",
    "ONNET3v2": ON.PATH.MODEL_DIR + "ONNET3v2/2019-03-24-00:22:37-a412/weights.h5",
    "ONNET": ON.PATH.MODEL_DIR + "ONNET/pretrained/weights.h5",
    "ONNET-LSTM": ON.PATH.MODEL_DIR + "ONNET/pretrained-lstm/weights.h5",
}

LINE_BLACKLIST = ["h02-037-02",  # corrupted stroke
                  "a02-062-01",  # 1263
                  "a02-062-02",  # 1076
                  "a02-062-03",  # 1212
                  "e05-265z-04",  # 1188
                  'g04-060-02',  # 1063
                  'j06-278z-04',  # 1588
                  'h04-141z-03']  # 828
