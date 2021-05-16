from dataset import MXFaceDataset
from paddle.io import DataLoader
from config import config as cfg
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter
from utils.utils_amp import MaxClipGradScaler
import paddle.nn.functional as F
from paddle.nn import ClipGradByNorm
from visualdl import LogWriter
import paddle
import backbones
import argparse
import losses
import time
import os


__Author__ = 'Quanhao Guo'
__Date__ = '2021.04.24.16.23'


def main(args):
    world_size = int(1.0)
    rank = int(0.0)
    local_rank = args.local_rank
    
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    else:
        time.sleep(2)
    
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    else:
        time.sleep(2)
    
    writer = LogWriter(logdir=cfg.logdir)
    
    trainset = MXFaceDataset(root_dir=cfg.rec)
    train_loader = DataLoader(dataset=trainset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)

    dropout = 0.4 if cfg.dataset == "webface" else 0
    backbone = eval("backbones.{}".format(args.network))(False, dropout=0.5, fp16=False)
    backbone.train()

    clip_by_norm = ClipGradByNorm(5.0)
    margin_softmax = eval("losses.{}".format(args.loss))()

    module_partial_fc = PartialFC(
        rank=0, local_rank=0, world_size=1, resume=0,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    scheduler_backbone = paddle.optimizer.lr.LambdaDecay(learning_rate=cfg.lr / 512 * cfg.batch_size, lr_lambda=cfg.lr_func, verbose=True)
    opt_backbone = paddle.optimizer.SGD(
        parameters=backbone.parameters(),
        learning_rate=scheduler_backbone,
        weight_decay=cfg.weight_decay,
        grad_clip=clip_by_norm)
    scheduler_pfc = paddle.optimizer.lr.LambdaDecay(learning_rate=cfg.lr / 512 * cfg.batch_size, lr_lambda=cfg.lr_func, verbose=True)
    opt_pfc = paddle.optimizer.SGD(
        parameters=module_partial_fc.parameters(),
        learning_rate=scheduler_pfc,
        weight_decay=cfg.weight_decay,
        grad_clip=clip_by_norm)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: 
        print("Total Step is: %d" % total_step)
    
    callback_verification = CallBackVerification(2000, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(100, rank, total_step, cfg.batch_size, world_size, writer)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_scaler = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    for epoch in range(start_epoch, cfg.num_epoch):
        for step, (img, label) in enumerate(train_loader):
            label = label.flatten()
            global_step += 1
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            if cfg.fp16:
                scaled = grad_scaler.scale(x_grad)
                (features.multiply(scaled)).backward()
                grad_scaler._unscale(opt_backbone)
                grad_scaler.minimize(opt_backbone, scaled)
            else:
                (features.multiply(x_grad)).backward()
                opt_backbone.step()
            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.clear_gradients()
            opt_pfc.clear_gradients()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_scaler)
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step()
        scheduler_pfc.step()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='iresnet50', help='backbone network')
    parser.add_argument('--loss', type=str, default='ArcFace', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args = parser.parse_args()
    main(args)

