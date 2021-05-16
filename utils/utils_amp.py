import paddle
from paddle.amp import GradScaler


__Author__ = 'Quanhao Guo'
__Date__ = '2021.04.24.16.21'


class MaxClipGradScaler(GradScaler):
    def __init__(self, init_loss_scaling, max_scale: float, incr_every_n_steps=100):
        GradScaler.__init__(self, init_loss_scaling=init_loss_scaling, incr_every_n_steps=incr_every_n_steps)
        self.get_scale = 1.0
        self.max_scale = max_scale

    def scale_clip(self):
        if self.get_scale == self.max_scale:
            self._incr_ratio = 1.0
        elif self.get_scale < self.max_scale:
            self._incr_ratio = 2.0

    def scale(self, outputs):
        if not self._enable:
            return outputs
        self.scale_clip()
        # Short-circuit for the common case.
        if isinstance(outputs, paddle.Tensor):
            assert self._init_loss_scaling is not None
            return outputs * self._init_loss_scaling