import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.backend import ctc_decode
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import sparse_ops, math_ops, array_ops
from hwr.decoding.mlf import label2txt
from tqdm import tqdm


def best_path_raw(rnn_out):
    ret = []
    for i in range(len(rnn_out)):
        ret.append([np.argmax(row) for row in rnn_out[i]])
    return label2txt(ret, remove_dup=True)


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


