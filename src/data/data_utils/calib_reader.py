
import numpy as np

class CalibReader:

    def __init__(self, calib_path):
        self.calib_path = calib_path

    def read_calib(self):
        lines = []
        with open(self.calib_path) as calib_file:
            lines = calib_file.readlines()
        lines = list(filter(lambda x: len(x.split()) > 0, lines))
        calib_data = dict(list(map(lambda x: (x.split()[0][:-1], np.array(list(map(float, x.split()[1:])))), lines)))
        return calib_data