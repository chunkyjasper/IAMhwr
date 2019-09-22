import io
import re
from collections import defaultdict

import dill as pickle
from nltk.lm import NgramCounter
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.util import ngrams
from tqdm import tqdm

from hwr.constants import PATH


def clean_space(txt):
    return re.sub(r"\s+", " ", txt)


def clean_newline(txt):
    return re.sub(r"\n", " ", txt)


def clean_chars(txt):
    return re.sub(r"[^!\"&'/\(\)\[\]*,\-+.\s:;?0-9a-zA-Z]", "", txt)


def clean_text(txt):
    return clean_space(clean_newline(clean_chars(txt))).lower()


# Given
def update_counter(counter, ngram, fname, batch=10000):
    print("Updating counter with file:")
    print(fname)
    with io.open(fname, encoding='utf8') as fin:
        txt = fin.read()
        txt = clean_text(txt)
    # n of batches
    k = int(len(txt) / batch)

    # for each ngram
    print("Updating ngrams:")
    for n in tqdm(range(1, ngram + 1)):
        # for each batch
        for i in range(k + 1):
            # update ngram
            start = 0 if i == 0 else i * batch - n + 1
            last = -1 if i == k else (i + 1) * batch + n - 1
            counter.update([ngrams(txt[start:last], n)])
    return counter


# Overriding ConditionalFreqDist constructor
def init_override(self, cond_samples=None, with_freq=False):
    defaultdict.__init__(self, FreqDist)
    if cond_samples:
        if not with_freq:
            for (cond, sample) in cond_samples:
                self[cond][sample] += 1
        else:
            for (cond, sample, freq) in cond_samples:
                self[cond][sample] += freq


def prune_cond_dist(cond_dist, threshold=10):
    setattr(ConditionalFreqDist, '__init__', init_override)
    tuple_list = list(cond_dist.__reduce__()[4])
    cond_samples = []
    for cond, freq_dist in tuple_list:
        for c in freq_dist:
            n = freq_dist[c]
            if n > threshold:
                cond_samples.append((cond, c, n))
    return ConditionalFreqDist(cond_samples=cond_samples, with_freq=True)


def prune_counter(counter, order, threshold=10):
    new_counter = NgramCounter()
    new_counter._counts[1] = counter[1]
    for i in range(2, order+1):
        new_counter._counts[i] = prune_cond_dist(counter[i], threshold=threshold)
    return new_counter


ngram = 7
fname = "lm_7gram_counter.pkl"

if __name__ == "__main__":
    counter = NgramCounter()
    for p in range(1, 100):
        print("file {}".format(p))
        fnum = "0000" + str(p) if p < 10 else "000" + str(p)
        fn = PATH.BASE_DIR + '../data/1blm/training-monolingual.tokenized.shuffled/news.en-' + fnum + '-of-00100'
        counter = update_counter(counter, ngram, fn)

    with open(fname, 'wb') as fout:
        pickle.dump(counter, fout)

    print("Completed.")
