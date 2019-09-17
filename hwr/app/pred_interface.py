import abc
from hwr.data.datarep import Point, PointSet
from hwr.constants import ON
import numpy as np
from hwr.models.ONNET import ONNET


# Implement and pass to draw area constructor for prediction algorithm.
class IPred(object):
    def __init__(self):
        __metaclass__ = abc.ABCMeta

    # Take a list of (list of (x,y)) and return features
    @abc.abstractmethod
    def get_features(self, coordinates):
        return

    # predict top n output with ^ return value
    @abc.abstractmethod
    def predict(self, features, n):
        return


class ONNETpred(IPred):

    def __init__(self):
        super().__init__()
        self.model = ONNET(preload=True)

    def get_features(self, strokes):
        points = []
        for i in range(len(strokes)):
            stroke = strokes[i]
            for x, y in stroke:
                points.append(Point(i, 0, x, y))
        pointset = PointSet(points=points)
        #pointset.plot_strokes()
        scheme = ON.PREPROCESS.SCHEME6
        pointset.preprocess(**scheme)
        #pointset.plot_strokes()
        print(pointset)
        return pointset.generate_features(add_pad=10)

    def predict(self, features, n):
        x = np.expand_dims(features, axis=0)
        results = self.model.predict(x, top=n)[0]
        print(results)
        return results





