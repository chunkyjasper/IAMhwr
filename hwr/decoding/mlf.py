import itertools

import numpy as np

from hwr.constants import ON

"""
Conversion between text, label and MLF
"""


# map operation for ndarray
def ndarray_map(f, a, as_list=False):
    if isinstance(a, np.ndarray):
        a = a.tolist()
    ret = list(map(f, a))
    return ret if as_list else np.array(ret)


def txt2label(txt, multiple=False):
    if multiple:
        return ndarray_map(lambda x: txt2label(x), txt)
    label = np.ones(len(txt)) * -1
    txt = list(map(lambda x: ON.DATA.CHARS.index(x), txt))
    for i in range(len(txt)):
        label[i] = txt[i]
    return label


def label2txt(labels, remove_dup=False, multiple=False):
    if multiple:
        return ndarray_map(lambda x: label2txt(x, remove_dup=remove_dup), labels)
    if remove_dup:
        labels = list(ch for ch, _ in itertools.groupby(labels))
    txt = list(map(lambda x: '' if x == -1 or x == ON.DATA.BLANK_IDX else ON.DATA.CHARS[int(x)], labels))
    txt = ''.join(txt)
    return txt


def label2mlf(labels, remove_dup=False, multiple=False):
    if multiple:
        return ndarray_map(lambda x: label2mlf(x, remove_dup=remove_dup), labels)
    if remove_dup:
        labels = list(ch for ch, _ in itertools.groupby(labels))
    mlf = list(map(lambda x: '' if x == -1 or x == ON.DATA.BLANK_IDX else ON.DATA.CHARS_MLF[int(x)], labels))
    mlf = [x for x in mlf if x != '']
    return mlf


def mlf2label(mlf, multiple=False):
    if multiple:
        return ndarray_map(lambda x: mlf2label(x), mlf)
    label = np.zeros(len(mlf))
    mlf_idx = list(map(lambda x: ON.DATA.CHARS_MLF.index(x), mlf))
    for i in range(len(mlf_idx)):
        label[i] = mlf_idx[i]
    return label


def mlf2txt(mlf, multiple=False):
    if multiple:
        return ndarray_map(lambda x: mlf2txt(x), mlf)
    return label2txt(mlf2label(mlf))





