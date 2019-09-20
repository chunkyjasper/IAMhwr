import abc

import dill as pickle
import numpy as np
from nltk.lm import Vocabulary
from tensorflow.keras import backend as K
from tensorflow.keras.backend import ctc_decode
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import sparse_ops, math_ops, array_ops
from tqdm import tqdm

from hwr.constants import ON
from hwr.decoding.mlf import label2txt
from hwr.decoding.trie import Trie
from hwr.decoding.trie_beam_search import trie_beam_search
from hwr.lm.lm import KneserNeyBackoff


# Interface for a CTC decoding algorithm
class ICTCDecoder:
    def __int__(self):
        __metaclass__ = abc.ABCMeta

    """
    Given the softmax output of RNN of size (batch_size, time_step, labels)
    return top n predictions for each item in batch.
    Output have type list(predictions) of length = batch_size,
    where predictions is list(string) of length = top_n
    """
    @abc.abstractmethod
    def decode(self, rnn_out, top_n):
        return


class BestPathDecoder(ICTCDecoder):
    def __init__(self):
        super(BestPathDecoder, self).__init__()

    def decode(self, rnn_out, top_n):
        pred = best_path(rnn_out)
        return list(map(lambda p: [p for _ in range(top_n)], pred))


class BeamSearchDecoder(ICTCDecoder):
    def __init__(self, beam_width):
        super(BeamSearchDecoder, self).__init__()
        self.beam_width = beam_width

    def decode(self, rnn_out, top_n):
        return beam_search(rnn_out, self.beam_width, top_paths=top_n)


# See trie_beam_search.py
class TrieBeamSearchDecoder(ICTCDecoder):
    def __init__(self, beam_width, lm=None, trie=None, lm_order=0):

        super(TrieBeamSearchDecoder).__init__()
        self.beam_width = beam_width
        self.lm = lm
        self.lm_order = lm_order
        if lm is None:
            self.lm_order = 5
            self.lm = load_lm(self.lm_order, ON.PATH.LM_DATA_DIR + "5gram_counter_pruned-100.pkl")
        self.trie = trie
        if trie is None:
            self.trie = load_trie(ON.PATH.LM_DATA_DIR + "wiki-100k.txt")

    def decode(self, rnn_out, top_n):
        return trie_beam_search(rnn_out, self.lm, self.trie,
                                self.beam_width, top_paths=top_n,
                                lm_order=self.lm_order)


def load_trie(file_path):
    with open(file_path, encoding='utf8') as fin:
        vocab_txt = fin.read().splitlines()
    # vocabs to lowercase and remove ones with illegal chars
    vocab_txt = [v.lower() for v in vocab_txt if v[:2] != "#!"]
    vocab_txt = [v for v in vocab_txt if all([c in ON.DATA.CHARS for c in v])]
    t = Trie()
    t.mass_insert(vocab_txt)
    return t


def load_lm(order, counter_file_path):
    with open(counter_file_path, 'rb') as fin:
        counter = pickle.load(fin)
    chars = Vocabulary(ON.DATA.CHARS)
    return KneserNeyBackoff(order, backoff=0.4, counter=counter, vocabulary=chars)


def best_path_raw(rnn_out, remove_dup=True):
    ret = []
    for i in range(len(rnn_out)):
        ret.append([np.argmax(row) for row in rnn_out[i]])
    return label2txt(ret, remove_dup=remove_dup, multiple=True)


# Greedy search. Just pick the most probable candidate at each time step.
def best_path(rnn_out):
    result_list, _ = ctc_decode(rnn_out, np.ones(rnn_out.shape[0]) * rnn_out.shape[1])
    result_list = K.eval(result_list[0])
    pred = label2txt(result_list, multiple=True)
    return list(pred)


# Beam search. Keep track of the best n 'beam' per timestep
# And calculate the prob of them to find the most probable sequence.
def beam_search(rnn_out, beam_width, top_paths=1):
    _EPSILON = 1e-7
    num_of_samples = rnn_out.shape[0]
    input_length = np.ones(num_of_samples) * rnn_out.shape[1]
    input_length = math_ops.to_int32(input_length)
    rnn_out = math_ops.log(array_ops.transpose(rnn_out, perm=[1, 0, 2]) + _EPSILON)

    decoded, log_prob = ctc.ctc_beam_search_decoder(
        inputs=rnn_out,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths,
        merge_repeated=False)

    decoded_dense = [
        sparse_ops.sparse_to_dense(
            st.indices, st.dense_shape, st.values, default_value=-1)
        for st in decoded
    ]
    candidates = [K.eval(i) for i in tqdm(decoded_dense)]

    pred = [[] for _ in range(num_of_samples)]
    for k in range(num_of_samples):
        for c in candidates:
            pred[k].append(c[k])
    pred = [list(label2txt(p, multiple=True)) for p in pred]
    return pred


