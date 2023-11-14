import os
from data.videodata_online_gaussian import VIDEODATA_ONLINE_GAUSSIAN


class REDS_BLURDOWN_GAUSSIAN(VIDEODATA_ONLINE_GAUSSIAN):
    def __init__(self, args, name='REDS_BlurDown_Gaussian', train=True):
        super(REDS_BLURDOWN_GAUSSIAN, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'HR')
        print("DataSet gt path:", self.dir_gt)
