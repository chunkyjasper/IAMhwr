from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Input, Dense, Activation, \
    CuDNNGRU, LSTM, \
    Lambda, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from hwr.constants import ON, PRETRAINED
from hwr.models.model import HWRModel


# New preprocessing: scheme 6
# Aim : to optimize onnet3 parameters emperically with new preprocessing strat
# 1- 3 Are attempts with triple 2 average pool(divide timestep by 8)
# Attempt 1: 100 200 300 400 5 4 3 2 cnn 300 600 rnn 14.79cer
# Attempt 2: Same but adaptive padding 13.86cer
# Attempt 3: Max Pool 15.09
# Attempt 4: ONNET3 with new preprocessed data. Does not converge
# 5-6 are attempts with double 2 average pool(scale down timestep by 4)
# Attempt 5: 50 100 150 200 4 3 2 2 200 200. 10.31cer new best (2 average pool)
# Attempt 6: 50 100 150 200 4 3 2 2 200 200. 9.92 cer (trained with adaptive padding)
# Experiment with rnn size
# Attempt 7: 50 100 150 200 4 3 2 2 150 150. 16.91 loss 10.14cer
# Attempt 8: 50 100 150 200 4 3 2 2 100 100. 16.62 loss 10.48cer
# Attempt 9: 40 80  120 160 3 2 2 2 150 150  16.54 loss 10.27cer
# Reproduce 6 to confirm correctness
# Attempt10: 50 100 150 200 4 3 2 2 200 200 16.58 loss 10.80cer
# Attempt11: 40 60  80  100 3 2 2 2 128 128 17.35 loss 11.5cer
# 3 -> 2 pooling with attempt 6 config
# Attempt12: 50 100 150 200 4 3 2 2 200 200. 19.06 loss 11.92cer(fast)
# single 2 pooling with attempt 6 config
# Attempt13: 50 100 150 200 4 3 2 2 200 200. ///
# Convolution pooling(pool size 2 stride 2) to replace average pool
# Attempt14: 50 100 150 200 4 3 2 2 200 200. 17.58 loss 10.81cer
# Attempt15: 50 100 150 200 4 2 3 4(stride 2) 200 200. 16.41 loss 10.4cer
# Attempt16: 50 100 150 200 4 3 2 7(stride 2) 200 200. 20.2 loss 20.4cer
# Return to average pooling
# Attempt17: 64 128 256 256 4 3 2 2 128 128 16.3 loss 9.72cer new best
# Attempt18: 128 256 512 512 4 3 2 2 128 128 16.6 loss 10.9 cer
# Attempt19: 64 128 128a2 256 256a2 4 3 2 2 2 128 128 15.5 loss 9.82cer
# Attempt20: 64 128a2 128 256 256a2 3 3 2 2 2 128 128 15.6 loss 9.20 cer new best
# Attempt21: 64 64 128a2 128 256 256a2 2 2 2 2 2 2 128 128 16.8 loss
# Attempt22: 64 128a2 128 256 256a2 3 3 2 2 2 128 256 16.38 loss 10.2cer
# Attempt23: 64 128a2 128 256 256a2 3 3 2 2 2 64 128 does not converge
# Attempt24: 64 128a2 128 256 256a2 3 3 2 2 2 128 does not look good so stop
# Attempt25: LSTM of 20. 15.6 loss 9.35cer 9.91 cer for early stopped
# Attempt26: Maxpool of 20. Does not converge
# Attempt27: Average pool of 3 for first of 20. 10.7
# Attempt28: 20, but 128 1d conv at the end. 9.95
# Attempt29: A20 with preprocess5. 16.0loss 9.45cer
# Attempt30: A30 but 128 5 for first layer. 15.9loss 9.91cer
# Attempt31: 64 128a2 128 256 256a2 5 3 3 2 2 100 100 9.70cer 15.8 loss
# Attempt32: a20 but 100 100. Very slow to converge
# Attempt33: a20 but add a 32 3 layer at front. Fast training, 15.8 loss 9.47 cer
# Attempt34: 32 64 96a2 128 160 192a2, 96 96. Scheme 6. 15.5 loss 9.82cer
# Attempt35: 32 64 96a2 128 160 200a2, 100 100. Scheme 6. 10.4cer batch 30
# Attempt20h: a20 half size. 16.5 loss 10.0 cer
# Attempt20h2: a20, 50 100 100 200 200 100 100 10.1cer 16.2loss
# Attempt20h3: a20, 75 150 150 300 300 150 150 15.3loss 9.22cer
# ATTEMPT20h4: 20h3 with extra 37 at first 9.43cer
# Attempet20h5: a20 but 128 1 at last and 64 64. 9.70cer
# Attempt20h6: a20 but 1 layer gru. 12.0cer
# Attempt20h7 170. 15.8loss 9.86cer
# Attempt36: 32 64 64 128 128 256 256 5 3 3 3 2 2, 32pool, 128 128 gru. 9.71 cer
# Attempt37: Pre6 32 64 96 96 128 128 5 3 3 2 2 1 22pool, 64 64. 15.7 loss, 9.60cer Effcient, 250k param
# Attempt38: Pre6 48 72 96 120 144 168 5 3 3 2 2 1, 84 84. 9.19cer new best! 9.34 try2
# Attempt38h1: 64 96 128 128 160 192 5 3 3 2 2 1, 96 96. 9.32cer try2 loss 9.18
# Attempt38h2: 128 128. 9.46 cer
# Switching optimzer. Adam -> sgd 1e-5 -> 1e-6
# Attempt39: 60 80 100 100 130 160 80 80 5 3 3 2 2 1. 14.68 loss 8.82cer (396) adam sgd 1e-4 e-5 e-6
# Attempt40: 60 80 100 100 130 160 80 80 7 5 3 3 2 2. 14.46 loss 8.64cer (402)
# Attempt41: 60 80 100 100 130 160 80 80 7 5 5 3 3 3. 14.29 loss 8.50cer (412) beam search 8.37
# Attempt42: 60 80 100 100 130 160 80 80 9 7 7 5 5 3  14.70 loss 8.75cer
# Attempt43: 80 100 120 120 140 160 80 80 9 7 5 3 3 3. 14.39 loss 9.09cer
# Attempt44: 80 100 120 140 160 200 100 100 7 5 5 3 3 3. 14.13loss 8.43cer new best
# Attempt45: 60 80 100  100 130 160 80 80 9 7 7 5 3 3, 3 2 pool. 16.5loss, 9.74cer
# Attempt46: 60 90 120 120 160 200 100 100 7 5 5 3 3 3. 13.98 loss 8.45 cer

# Implementation of model
class ONNET(HWRModel):
    def __init__(self, preload=True, gru=False):
        if gru:
            self.rnn = CuDNNGRU
            super(ONNET, self).__init__(preload=preload)
        else:
            self.rnn = LSTM
            super(ONNET, self).__init__()
            if preload:
                self.load_weights(PRETRAINED['ONNET-LSTM'], full_path=True)

    def get_prediction_layer(self):
        return "softmax"

    def get_input_layer(self):
        return "xs"

    def get_optimizer(self):
        return SGD(lr=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        #return 'adam'

    def get_loss(self):
        return {'ctc': lambda y_true, y_pred: y_pred}

    def get_model_conf(self):
        input_shape = (None, 6)
        inputs = Input(shape=input_shape, dtype='float32', name='xs')
        inner = inputs

        inner = cnn_bn_relu(inner, 60, 7)
        inner = cnn_bn_relu(inner, 80, 5)
        inner = cnn_bn_relu(inner, 100, 5)
        inner = AveragePooling1D(pool_size=2)(inner)

        inner = cnn_bn_relu(inner, 100, 3)
        inner = cnn_bn_relu(inner, 130, 3)
        inner = cnn_bn_relu(inner, 160, 3)

        inner = AveragePooling1D(pool_size=2)(inner)

        # No significant difference between gru and lstm
        inner = self.bi_gru(inner, 80)
        inner = self.bi_gru(inner, 80)

        inner = BatchNormalization()(inner)

        inner = Dense(ON.DATA.CHARS_SIZE, kernel_initializer='he_normal')(inner)
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

    def bi_gru(self, inner, hidden_unit):
        gru_1a = self.rnn(hidden_unit, return_sequences=True,
                      kernel_initializer='he_normal')(inner)
        gru_1b = self.rnn(hidden_unit, return_sequences=True,
                      go_backwards=True, kernel_initializer='he_normal')(inner)
        gru1_merged = concatenate([gru_1a, gru_1b])
        return gru1_merged


def cnn_bn_relu(inner, filters, kernel_size):
    inner = Conv1D(filters, kernel_size, padding="same", kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    return inner
