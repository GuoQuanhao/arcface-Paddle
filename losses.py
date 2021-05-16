import paddle
from paddle import nn


__Author__ = 'Quanhao Guo'
__Date__ = '2021.04.24.16.22'


class CosFace(nn.Layer):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        m_hot = paddle.nn.functional.one_hot(label.astype('long'), num_classes=85742) * self.m
        cosine -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Layer):
    def __init__(self, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: paddle.Tensor, label):
        m_hot = paddle.nn.functional.one_hot(label.astype('long'), num_classes=85742) * self.m
        cosine = cosine.acos()
        cosine += m_hot
        cosine = cosine.cos() * self.s
        return cosine
