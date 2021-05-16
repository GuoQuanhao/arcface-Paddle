import paddle
import sys
import argparse
sys.path.append("..")
import backbones
from utils.utils_callbacks import CallBackVerification
import re


__Author__ = 'Quanhao Guo'
__Date__ = '2021.04.24.11.37'


candidate_models = ['resnet_face18_fp16.pdparams', 'resnet_face34_fp16.pdparams', 'resnet_face50_fp16.pdparams', 'resnet_face100_fp16.pdparams']

def main(args):
    '''
    For the CallBackVerification class, you can place you val_dataset,
    like ["lfw"], also you can use ["lfw", "cplfw", "calfw"], my cpu is
    pool ,so I use ["lfw"] for only one val_dataset.
    
    For the callback_verification function, the batch_size must be divisible by 12000!
    Cause the length of dataset is 12000.
    '''
    backbone = eval("backbones.{}".format(args.network))()
    res_num = re.sub("\D", "", args.network)
    for candidate_model in candidate_models:
        if res_num in candidate_model:
            model_params = candidate_model
            break
    print('INFO:' + args.network + ' chose! ' + model_params + ' loaded!')
    state_dict = paddle.load(model_params)
    backbone.set_state_dict(state_dict)
    callback_verification = CallBackVerification(1, 0, ["lfw", "cplfw", "calfw"], "../faces_emore")
    callback_verification(1, backbone, batch_size=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paddle ArcFace Testing')
    parser.add_argument('--network', type=str, default='iresnet50', help='backbone network')
    args = parser.parse_args()
    main(args)