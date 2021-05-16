from utils.utils_callbacks import CallBackVerification
from backbones import *
import torch
import pickle
import paddle


__Author__ = 'Quanhao Guo'
__Date__ = '2021.04.24.11.05'


# read the torch model & save as pickle
# you need input you torch model path
# for example, I translate the 'backbone_r50_fp16.pth'
state_dict = torch.load('backbone_r50_fp16.pth')
npy = [value.cpu().numpy() for key, value in state_dict.items()]
with open('iresnet_50.pkl', 'wb') as fp:
    result = pickle.dump(npy, fp)

# read the pickle and save as paddle model
with open('iresnet_50.pkl', 'rb') as file:
    weight = pickle.load(file)

new_weight = []
for i in weight:
    if len(i.shape) != 0:
        new_weight.append(i)

backbone = iresnet50()
paddle.save(backbone.state_dict(), 'resnet_50_fp16.pdparams', 3)
state_dict = paddle.load('resnet_50_fp16.pdparams')

i = 0
for key, value in state_dict.items():
    if state_dict[key].shape==new_weight[i].shape:
        state_dict[key]=new_weight[i]
    else:
        state_dict[key]=new_weight[i].T
    i += 1

backbone.set_state_dict(state_dict)

paddle.save(backbone.state_dict(), 'paddle_pretrainedmodel/resnet_face50_fp16.pdparams', 3)


