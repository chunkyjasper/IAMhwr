import editdistance

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



