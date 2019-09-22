from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from hwr.constants import DATA


# rnn_out has dimension [batch_size, timestep, num_of_chars]
def trie_beam_search(rnn_out, lm, trie, bw, top_paths, lm_order, candidate_cap=5):
    return [__trie_beam_search(x, lm, trie, bw, top_paths, lm_order, candidate_cap) for x in tqdm(rnn_out)]


class Beam:
    def __init__(self, root):
        self.root = root
        self.curr_node = root
        self.text = ""

    def end_in_non_alpha(self):
        return self.text == "" or self.text[-1] in DATA.NON_ALPHAS

    def is_word(self):
        return self.curr_node.end

    def get_candidates(self):
        candidates = self.curr_node.get_children_chars()
        if self.is_word() or self.end_in_non_alpha():
            candidates += [c.upper() for c in candidates]
            candidates += DATA.NON_ALPHAS
        candidates += "%"
        return candidates

    def extend(self, c):
        new_beam = Beam(self.root)
        if c in DATA.NON_ALPHAS:
            new_beam.curr_node = self.root
        else:
            new_beam.curr_node = self.curr_node.children[c.lower()]
        new_beam.text = self.text + c
        return new_beam


def __trie_beam_search(mat, lm, trie, bw, top_paths, lm_order, candidate_cap):
    # pb[t][beam]: P of {beam} at time {t} ending with blank '%'
    # pnb[t][beam]: P of {beam} at time {t} ending with any non blank chars
    # Ptxt[beam] : P of {beam} given a language model.
    pb, pnb, ptxt = defaultdict(Counter), defaultdict(Counter), defaultdict()
    timestep, chars_size = mat.shape
    # add a time step 0 for P(t-1) at t=1
    mat = np.vstack((np.zeros(chars_size), mat))
    empty_beam = Beam(trie.root)
    pb[0][empty_beam] = 1
    pnb[0][empty_beam] = 0
    ptxt[empty_beam] = 1
    beams_prev = [empty_beam]

    non_alphas = DATA.NON_ALPHAS
    letters = DATA.CHARS

    for t in range(1, timestep + 1):
        for beam in beams_prev:
            # Get candidates by looking at trie
            candidates = beam.get_candidates()
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
                    l_plus = beam.text + char
                    letter_idx = letters.index(char)

                    new_beam = next((b for b in beams_prev if b.text == l_plus), None)
                    if not new_beam:
                        new_beam = beam.extend(char)

                    # Apply character level language model and calculate Ptxt(beam)
                    prefix = beam.text[-(lm_order - 1):]
                    if lm:
                        # Ptxt(beam+c) = P(c|last n char in beam)
                        ptxt[new_beam] = lm.score(char.lower(), [p for p in prefix.lower()])
                    else:
                        ptxt[new_beam] = 1

                    # if new candidate and last char in the beam is same
                    if len(beam.text) > 0 and char == beam.text[-1]:
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

    return [b.text for b in beams_prev[:top_paths]]
