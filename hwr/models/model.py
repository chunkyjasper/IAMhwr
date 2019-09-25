import abc
import datetime
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

from hwr.constants import PRETRAINED, DATA, PATH
from hwr.decoding.ctc_decoder import TrieBeamSearchDecoder
from hwr.models.metrics import character_error_rate, word_error_rate


# Interface for prediction model
class HWRModel(object):
    def __init__(self, chars=DATA.CHARS, preload_key=None,
                 decoder=None):
        __metaclass__ = abc.ABCMeta
        self.decoder = decoder
        if decoder is None:
            self.decoder = TrieBeamSearchDecoder(beam_width=25)
        self.chars = chars
        self.class_name = type(self).__name__
        self.ckptdir = PATH.CKPT_DIR + self.class_name + "/"
        self.char_size = len(chars) + 1
        self.model = self.get_model_conf()
        self.compile()
        if preload_key:
            self.pretrained = PRETRAINED[preload_key]
            print("preloading model weights from {}".format(self.pretrained))
            self.load_weights(self.pretrained, full_path=True)

    @abc.abstractmethod
    def get_model_conf(self):
        return

    @abc.abstractmethod
    def get_prediction_layer(self):
        return

    @abc.abstractmethod
    def get_input_layer(self):
        return

    @abc.abstractmethod
    def get_optimizer(self):
        return

    @abc.abstractmethod
    def get_loss(self):
        return

    def get_intermediate_model(self, layer_name):
        in_model = Model(inputs=self.model.get_layer(self.get_input_layer()).output,
                         outputs=self.model.get_layer(layer_name).output)
        # dummy loss and optimizer, predict with Sequence class requires compiled
        in_model.compile(loss={layer_name: lambda y_true, y_pred: y_pred}, optimizer='adam')
        return in_model

    def get_pred_model(self):
        return self.get_intermediate_model(self.get_prediction_layer())

    def train(self, train_seq, test_seq, epochs=100, earlystop=5):
        ckptdir = self.ckptdir + get_time() + '/'
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(ckptdir + 'weights.h5',
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=1)
        es_callback = tf.keras.callbacks.EarlyStopping(patience=earlystop)
        self.model.fit_generator(
            generator=train_seq,
            validation_data=test_seq,
            shuffle=True,
            verbose=1,
            epochs=epochs,
            callbacks=[cp_callback, es_callback]
        )

    def predict_softmax(self, x):
        if isinstance(x, Sequence):
            sm = self.get_pred_model().predict_generator(x, verbose=1)
        else:
            sm = self.get_pred_model().predict(x, verbose=1)
        return sm

    # return top n predicted text.
    def predict(self, x, decoder=None, top=1):
        if decoder is None:
            decoder = self.decoder
        softmaxs = self.predict_softmax(x)
        pred = decoder.decode(rnn_out=softmaxs, top_n=top)
        if top == 1:
            try:
                pred = [p[0] for p in pred]
            except IndexError:
                print("Index Error: {}".format(pred))
        return pred

    def evaluate(self, eval_seq, metrics=None, decoder=None):
        if metrics is None:
            metrics = [character_error_rate, word_error_rate]
        if decoder is None:
            decoder = self.decoder
        _, y_true = eval_seq.get_xy()
        y_pred = self.predict(eval_seq, decoder=decoder)
        ret = {}
        for m in metrics:
            ret[m.__name__] = m(y_true, y_pred)
        return ret

    # Keras cannot save custom loss and keras optimizer, so have to recompile after loading
    def compile(self):
        self.model.compile(loss=self.get_loss(),
                           optimizer=self.get_optimizer())

    def save_weights(self, file_name="", full_path=False):
        if not file_name:
            file_name = get_time() + '.h5'
        if not full_path:
            file_name += self.ckptdir
        self.model.save_weights(file_name)

    def load_weights(self, file_name, full_path=False):
        if not full_path:
            file_name += self.ckptdir
        self.model.load_weights(file_name)
        self.compile()

    def get_model_summary(self):
        return self.model.summary()




# get timestamp
def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")



