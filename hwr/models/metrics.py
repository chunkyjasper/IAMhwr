import editdistance
import re


# edit distance/length of ground truth
def character_error_rate(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    total_cer = 0
    for i in range(len(y_true)):
        ed = editdistance.eval(y_true[i], y_pred[i])
        char = len(y_true[i])
        total_cer += ed / char
    avg_cer = total_cer / len(y_true)
    return avg_cer


# Edit distance of words
def word_error_rate(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    pattern = r'[\w]+'
    total_wer = 0
    for i in range(len(y_true)):
        gt, pred = y_true[i], y_pred[i]
        words_gt = re.findall(pattern, gt)
        words_pred = re.findall(pattern, pred)
        words = list(set(words_gt + words_pred))
        idx_gt = []
        for w in words_gt:
            idx_gt.append(words.index(w))
        idx_pred = []
        for w in words_pred:
            idx_pred.append(words.index(w))
        ed = editdistance.eval(idx_gt, idx_pred)
        wer = ed / len(idx_gt)
        total_wer += wer
    avg_wer = total_wer / len(y_true)
    return avg_wer


# Out of vocab rate given dict in trie
def oov(trie, lines):
    pattern = r'[\w]+'
    words = [re.findall(pattern, l) for l in lines]
    words = [item for sublist in words for item in sublist]
    is_word = [trie.is_word(w) for w in words]
    return 1 - (sum(is_word) / len(is_word))





