import os
from decimal import Decimal

from hwr.constants import ON, SPLIT
from hwr.data.datarep import PointSet, Point
from hwr.decoding.mlf import mlf2txt
import numpy as np
from lxml import etree


# A sample from the IAM online database
class Sample(object):
    def __init__(self, xml_path, ground_truth):
        self.xml_path = xml_path
        self.ground_truth = ground_truth
        self.__pointset = None

    def generate_features(self, preprocess=ON.PREPROCESS.CURRENT_SCHEME):
        return self.pointset.generate_features(preprocess=preprocess)

    # plot the sample
    def visualize(self):
        self.pointset.plot_samples()

    # Ground truth in readable form
    def get_ground_truth_text(self):
        return mlf2txt(self.ground_truth)

    # Read data in xml file into PointSet object
    @property
    def pointset(self):
        if self.__pointset is not None:
            return self.__pointset
        else:
            xml = open(self.xml_path, 'rb').read()
            root = etree.XML(xml)
            wbd, strokeset = root.getchildren()

            # Unpack white board description
            sl = wbd[0].attrib['corner']
            do = wbd[1].attrib['x'], wbd[1].attrib['y']
            vo = wbd[2].attrib['x'], wbd[2].attrib['y']
            ho = wbd[3].attrib['x'], wbd[3].attrib['y']

            # Unpack Strokes, return list of (stroke_id, time, x, y)
            strokes = []
            stroke_id = 1
            min_time = Decimal(strokeset.getchildren()[0].getchildren()[0].attrib['time'])
            for stroke in strokeset:
                for point in stroke:
                    t = (Decimal(point.attrib['time']) - min_time) * 1000
                    x = point.attrib['x']
                    y = point.attrib['y']
                    strokes.append([stroke_id, t, x, y])
                stroke_id += 1
            strokes = np.asarray(strokes, dtype=np.int)

            # Find the four edges of whiteboard in coordinate space
            r, b = do  # right, bottom edge
            l, _ = vo  # left edge
            _, u = ho  # upper edge
            r, b, l, u = int(r), int(b), int(l), int(u)

            # Move top left corner to origin then flip along y
            strokes[:, 2] = np.subtract(strokes[:, 2], l)
            strokes[:, 3] = np.subtract(strokes[:, 3], u)
            points = []
            for s in strokes:
                points.append(Point(*s))
        return PointSet(points=points, w=r - l, h=b - u, file_name=self.xml_path)


# load samples given the data directory and a split (e.g. train, test)
class IAMReader(object):

    def __init__(self, split, data_path=ON.PATH.DATA_DIR):
        self.data_path = data_path
        self.split = split
        self.samples = None

    # Given a data split, return the samples
    def get_samples(self):
        if self.samples is not None:
            return self.samples
        sample_names = []
        if self.split == SPLIT.ALL:
            all_split = [SPLIT.TRAIN, SPLIT.VAL1, SPLIT.VAL2, SPLIT.TEST]
            for split in all_split:
                sample_names += self.__get_sample_names_from_split(split)
        else:
            sample_names = self.__get_sample_names_from_split(self.split)
        self.samples = self.__get_samples_from_name(sample_names)
        return self.samples

    # Given a data split, return the sample names list
    def __get_sample_names_from_split(self, split):
        f = open(self.get_split_dir() + split)
        return [line.strip(' \n') for line in f]

    # Given samples name e.g. ['a02-050',], return samples with path to data and ground truth.
    def __get_samples_from_name(self, names, blacklist=ON.DATA.BLACKLIST):
        # File of ground truth of each sample
        f = open(self.__get_lines_data_dir() + "t2_labels.mlf")

        samples = []
        curr_path = ""
        curr_sample_name = ""
        curr_gt = []

        for line in f:
            # comments
            if line[0] == '#':
                continue
            # .lab file name
            # e.g. "/scratch/global/liwicki/wb/data/new-lang-model/transcriptions/a01-000u-05.lab"
            elif line[0] == '"':
                # Add the sample
                if curr_path and curr_sample_name in names:
                    curr_gt = curr_gt[:-1]
                    samples.append(Sample(curr_path, curr_gt))
                # Clear cached result
                curr_path = ""
                curr_sample_name = ""
                curr_gt = []
                # Read the next file name
                striped_line = line.strip(' "\n')
                line_split = striped_line.split('/')
                file_name = line_split[8].split('.')[0]
                if file_name in blacklist:
                    continue
                fn_split = file_name.split('-')
                path = fn_split[0] + "/" + fn_split[0] + "-" + fn_split[1][:3] + \
                       "/" + file_name + ".xml"
                path = ON.PATH.LINE_DATA_DIR + 'data/' + path

                # if corrupted file/not found, pass
                try:
                    if not os.path.getsize(path):
                        continue
                except FileNotFoundError:
                    # print("Missing file: {}".format(path))
                    continue

                curr_path = path
                curr_sample_name = fn_split[0] + "-" + fn_split[1]
            # Read the ground truth
            else:
                line_split = line.strip('\n')
                curr_gt.append(line_split)
        return samples



    def __get_lines_data_dir(self):
        return self.data_path + "lines/"

    def get_split_dir(self):
        return self.data_path + "split-config/"


def xmlpath2npypath(path, npz_dir):
    f_split = path.split('/')
    f_split[-4] = npz_dir
    f_split[-1] = f_split[-1][:-3] + 'npz'
    f = '/'.join(f_split)
    return f
