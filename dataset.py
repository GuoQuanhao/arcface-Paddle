from  paddle.io import Dataset
from paddle.vision import transforms
from mxnet_reader import recordio
from PIL import Image
from io import BytesIO
import numpy as np
import numbers
import paddle
import os


__Author__ = 'Quanhao Guo'
__Date__ = '2021.04.24.16.23'


class MXFaceDataset(Dataset):
    def __init__(self, root_dir):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = paddle.to_tensor(label, dtype='int64')
        # sample = image.imdecode(img).asnumpy()
        sample = np.array(Image.open(BytesIO(img)))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
