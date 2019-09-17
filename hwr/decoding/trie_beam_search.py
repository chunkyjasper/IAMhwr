from collections import defaultdict, Counter

import dill as pickle
import numpy as np
from tqdm import tqdm
from hwr.constants import ON, BASE_DIR
from hwr.lm.lm import KneserNeyBackoff
from nltk.lm import Vocabulary
from hwr.decoding.trie import Trie

# Cache models initialization
lm = None
trie = None


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


# get the ending alphabets given a word beam
def get_ending_alphas(text):
    end_alphas = ""
    for i in reversed(range(len(text))):
        if text[i].isalpha():
            end_alphas = text[i] + end_alphas
        else:
            break
    return end_alphas


# sm has dimension [sample, timestep, num_of_chars]
def trie_beam_search(rnn_out, bw, top_paths, use_lm=True, candidate_cap=5):
    return [__trie_beam_search(x, bw, top_paths, use_lm, candidate_cap) for x in tqdm(rnn_out)]


def __trie_beam_search(mat, bw, top_paths, use_lm, candidate_cap):
    global lm
    global trie
    lm_order = 5
    if lm is None:
        lm = load_lm(lm_order, BASE_DIR + '../data/lm/lm_5gramchar_counter_pruned-100.pkl')
    if trie is None:
        trie = load_trie(BASE_DIR + '../data/wiki-100k.txt')

    # pb[t][beam]: P of {beam} at time {t} ending with blank '%'
    # pnb[t][beam]: P of {beam} at time {t} ending with any non blank chars
    # Ptxt[beam] : P of {beam} given a language model.
    pb, pnb, ptxt = defaultdict(Counter), defaultdict(Counter), defaultdict(lambda: None)
    timestep, chars_size = mat.shape
    # add a time step 0 for P(t-1) at t=1
    mat = np.vstack((np.zeros(chars_size), mat))
    pb[0][''] = 1
    pnb[0][''] = 0
    ptxt[''] = 1
    beams_prev = ['']

    non_alphas = ON.DATA.NON_ALPHAS
    letters = ON.DATA.CHARS

    for t in range(1, timestep + 1):
        for beam in beams_prev:
            # Get ending alphabet, try to form a word in the trie
            ending_alphas = get_ending_alphas(beam).lower()
            candidates = trie.get_char_candidates(ending_alphas)
            # Allow uppercase and non alphabets only when a word is form/ not being formed
            if trie.is_word(ending_alphas) or ending_alphas == "":
                candidates += [c.upper() for c in candidates]
                candidates += non_alphas
            candidates += "%"

            # Check only top n candidates for performance
            if len(candidates) > candidate_cap:
                candidates = sorted(candidates, key=lambda c: mat[t][letters.index(c)], reverse=True)[:candidate_cap]

            for char in candidates:
                # if candidate is blank
                if char == '%':
                    # Pb(beam,t) += mat(blank,t) * Ptotal(beam,t-1)
                    pb[t][beam] += mat[t][-1] * (pb[t - 1][beam] + pnb[t - 1][beam])

                # if candidate is non-blank
                else:
                    new_beam = beam + char
                    letter_idx = letters.index(char)

                    # Apply character level language model and calculate Ptxt(beam)
                    prefix = beam[-(lm_order - 1):]
                    if use_lm:
                        # Ptxt(beam+c) = P(c|last n char in beam)
                        ptxt[new_beam] = lm.score(char.lower(), [p for p in prefix.lower()])
                    else:
                        ptxt[new_beam] = 1

                    # if new candidate and last char in the beam is same
                    if len(beam) > 0 and char == beam[-1]:
                        # Pnb(beam+c,t) += mat(c,t) * Pb(beam,t-1)
                        pnb[t][new_beam] += mat[t][letter_idx] * pb[t - 1][beam]
                        # Pnb(beam,t) = mat(c,t) * Pnb(beam,t-1)
                        pnb[t][beam] += mat[t][letter_idx] * pnb[t - 1][beam]
                    else:
                        # Pnb(beam+c,t) = mat(c,t) * Ptotal(beam,t-1)
                        pnb[t][new_beam] += mat[t][letter_idx] * (pb[t - 1][beam] + pnb[t - 1][beam])
        Ptotal_t = pb[t] + pnb[t]
        # sort by Ptotal * Ptxt
        sort = lambda k: Ptotal_t[k] * ptxt[k]
        # Top (bw) beams for next iteration
        beams_prev = sorted(Ptotal_t, key=sort, reverse=True)[:bw]

    return beams_prev[:top_paths]
