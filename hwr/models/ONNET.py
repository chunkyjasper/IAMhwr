# suppress tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Input, Dense, Activation, \
     LSTM, GRU, \
    Lambda, BatchNormalization
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from hwr.constants import DATA
from hwr.models.model import HWRModel





# Implementation of model
class ONNET(HWRModel):
    def __init__(self, preload=False, gru=False, decoder=None, gpu=False):
        if gpu:
            # Importing at file heading makes cpu version unable to init
            from tensorflow.keras.layers import CuDNNGRU, CuDNNLSTM
        if gru:
            self.rnn = CuDNNGRU if gpu else GRU
            preload_key = "ONNET-GRU" if preload else None
        else:
            self.rnn = CuDNNLSTM if gpu else LSTM
            preload_key = "ONNET-LSTM" if preload else None
        super().__init__(preload_key=preload_key, decoder=decoder)

    def get_prediction_layer(self):
        return "softmax"

    def get_input_layer(self):
        return "xs"

    def get_optimizer(self):
        return SGD(lr=1e-4, momentum=0.9, nesterov=True, clipnorm=5)
        #return 'adam'

    def get_loss(self):
        return {'ctc': lambda y_true, y_pred: y_pred}

    def get_model_conf(self):
        input_shape = (None, 6)
        inputs = Input(shape=input_shape, dtype='float32', name='xs')
        inner = inputs

        inner = tdnn_bn_relu(inner, 60, 7)
        inner = tdnn_bn_relu(inner, 90, 5)
        inner = tdnn_bn_relu(inner, 120, 5)
        inner = AveragePooling1D(pool_size=2)(inner)

        inner = tdnn_bn_relu(inner, 120, 3)
        inner = tdnn_bn_relu(inner, 160, 3)
        inner = tdnn_bn_relu(inner, 200, 3)

        inner = AveragePooling1D(pool_size=2)(inner)

        # No significant difference between gru and lstm
        inner = self.bi_rnn(inner, 60)
        inner = self.bi_rnn(inner, 60)
        inner = self.bi_rnn(inner, 60)
        inner = self.bi_rnn(inner, 60)


        inner = BatchNormalization()(inner)

        inner = Dense(DATA.CHARS_SIZE, kernel_initializer='he_normal')(inner)
        y_pred = Activation('softmax', name='softmax')(inner)

        # parameters for CTC loss, fed as network input
        labels = Input(name='ys',
                       shape=[None], dtype='float32')
        input_length = Input(name='ypred_length', shape=[1], dtype='int64')
        label_length = Input(name='ytrue_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.__ctc_lambda_func, output_shape=(1,),
                          name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[inputs, labels, input_length, label_length],
                      outputs=loss_out)
        return model

    def __ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def bi_rnn(self, inner, hidden_unit):
        rnn_a = self.rnn(hidden_unit, return_sequences=True,
                      kernel_initializer='he_normal')(inner)
        rnn_b = self.rnn(hidden_unit, return_sequences=True,
                      go_backwards=True, kernel_initializer='he_normal')(inner)
        rnn_merged = concatenate([rnn_a, rnn_b])
        return rnn_merged


def tdnn_bn_relu(inner, filters, kernel_size):
    inner = Conv1D(filters, kernel_size, padding="same", kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    return inner


def inception(filters, filter_size, pool_size, strides, inner):
    n = int(filters / (len(filter_size) + 1))
    inc = []
    for i in range(len(filter_size)):
        inc_this = Conv1D(n, filter_size[i], strides, padding="same", kernel_initializer='he_normal')(inner)
        inc.append(inc_this)
    inc_avg = AveragePooling1D(pool_size=pool_size, strides=strides, padding="same")(inner)
    inc_avg = Conv1D(n, 1, padding="same", kernel_initializer='he_normal')(inc_avg)
    inc.append(inc_avg)
    inner = concatenate(inc)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    return inner


def residual_inception(inner):
    inc1 = Conv1D(32, 1, padding="same", kernel_initializer='he_normal')(inner)
    inc2 = Conv1D(32, 1, padding="same", kernel_initializer='he_normal')(inner)
    inc2 = Conv1D(32, 3, padding="same", kernel_initializer='he_normal')(inc2)
    inc3 = Conv1D(32, 1, padding="same", kernel_initializer='he_normal')(inner)
    inc3 = Conv1D(32, 3, padding="same", kernel_initializer='he_normal')(inc3)
    inc3 = Conv1D(32, 3, padding="same", kernel_initializer='he_normal')(inc3)
    inc_cc = concatenate([inc1, inc2, inc3])
    inc_cc = Conv1D(256, 1, padding="same", kernel_initializer='he_normal')(inc_cc)
    inner = add([inner, inc_cc])
    return inner


# Override default LSTM activation function for CuDNN compatiblity
class LSTM(LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(recurrent_activation='sigmoid', *args, **kwargs)

