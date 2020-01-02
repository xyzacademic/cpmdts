import torch
import argparse
import numpy as np

import os
import torchvision.transforms as transforms

import torch.nn as nn

from dataloader import CPMDataset3, CropAndPad
# from segmenter import Brats19Segmenter
from classifier2 import Classifier
from lossfunction import DiceLoss, Brats19Loss, Brats19LossV2
from torch.utils.data import DataLoader
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='Train DTS with 2d segmentation')


parser.add_argument('--norm-axis', default='all', type=str, help='Normalization axis')
parser.add_argument('--data', default=0, type=str, help='Data source')
parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
parser.add_argument('--resume', action='store_true', help='Load pre-trained weights')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size.')
parser.add_argument('--seed', default=4096, type=int, help='Random seed.')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate.')
parser.add_argument('--epoch', default=100, type=int, help='The number of epochs')
parser.add_argument('--gpu', default=-1, type=int, nargs='+', help='Using which gpu.')
parser.add_argument('--wd', default=0.0000, type=float, help='Initial weight decay.')
parser.add_argument('--net', default='ours', type=str, help='which network')
parser.add_argument('--loss', default='diceloss', type=str, help='which loss')
parser.add_argument('--schedule', default='s1', type=str, help='Training schedule, s1, s2, s3')
parser.add_argument('--syncbn', action='store_true', help='Using synchronize batchnorm')
parser.add_argument('--basefilter', default=16, type=int, help='Batch size.')
parser.add_argument('--cpu', action='store_true', help='Using CPU')
parser.add_argument('--distributed', action='store_true', help='Whether using distributed training')
parser.add_argument('--flip', action='store_true', help='Whether using random flip')
parser.add_argument('--comment', default='', type=str, help='comment')
parser.add_argument('--target', default='wt', type=str, help='comment')
parser.add_argument('--outdir', default='output', type=str, help='output file dir')

args = parser.parse_args()


########################################################################################
# Global Flag
########################################################################################

config = {}

# Config setting
config['norm_axis'] = args.norm_axis
config['resume'] = True if args.resume else False
config['use_cuda'] = False if args.cpu else True
config['fp16'] = args.fp16
config['dtype'] = torch.float16 if config['fp16'] else torch.float32
config['syncbn'] = True if args.syncbn else False
config['gpu'] = args.gpu
config['batch_size'] = args.batch_size
config['seed'] = args.seed
config['schedule'] = args.schedule
config['lr'] = args.lr
config['wd'] = args.wd
config['loss'] = args.loss
config['epoch'] = args.epoch
config['lr_decay'] = np.arange(1, config['epoch'])
config['experiment_name'] = args.net
config['cpu'] = True if args.cpu else False
config['distributed'] = True if args.distributed else False
config['save_path'] = "%s" % args.comment
config['flip'] = True if args.flip else False
config['outdir'] = args.outdir

print(sys.argv)
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

########################################################################################
# Data setting
########################################################################################

data_path = os.path.join(config['outdir'], 'single_train_v15_240')

current_path = os.getcwd()
if not os.path.exists(config['experiment_name']):
    os.makedirs(config['experiment_name'])

config['save_path'] = os.path.join(config['experiment_name'], os.path.join(args.data, config['save_path']))

########################################################################################
# Pre-processing, dataset, data loader
########################################################################################

data_keyword = ['']
# data_keyword = ['_t1', '_flair', '_t1ce', '_t2']

target_keyword = ['seg', 'wt', 'tc', 'et', 'ed', 'ncr']

data_transform = transforms.Compose(
    [
        CropAndPad('Brats19', target='data', channel=len(data_keyword)),
        # Normalize(view=config['norm_axis']),
        # RandomIntensity(prob=0.5, shift=0.8),
    ]
)
# data_transform=None
# both_transform = transforms.Compose(
#     [
#         # RandomFlip(view=None, prob=0.5),
#         # RandomRotate(),
#         RandomCrop(format_='Brats19', shape=(160, 192, 144)),
#         PartialRandomIntensity(prob=0.5, shift=0.8),
#
#     ]
# )
path_transform = transforms.Compose(
    [
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])

path_transform2 = transforms.Compose(
    [
        transforms.CenterCrop(224),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
both_transform = None
inference_transform = transforms.Compose(
    [
        # Normalize(view=config['norm_axis']),
        CropAndPad('Brats19', target='data', channel=len(data_keyword)),

    ]
)
label_transform = None

test_list = pd.read_csv(os.path.join(config['outdir'], 'patient_id.csv'))['CPM_RadPath_2019_ID']


testset = CPMDataset3(root=data_path, outdir=config['outdir'],
                      patient_list=test_list, labels=False, nifti=True,
                      data_keyword=data_keyword,
                         target_keyword=target_keyword,
                         data_transform=inference_transform, path_transform=path_transform2,
                         target_transform=label_transform)

test_loader = DataLoader(dataset=testset, batch_size=config['batch_size'], shuffle=False,
                          num_workers=config['batch_size'], pin_memory=True)
########################################################################################
# Network setting
########################################################################################

from new_net import NewModel
net = NewModel(channel_2d=3, channel_3d=1, num_classes=3, num_node=10,)
########################################################################################
# Model initialization, loss function setting
########################################################################################
model = Classifier(config=config)
model.net_initialize(net)
save_path = os.path.join(config['save_path'], "checkpoints")

model.optimizer_initialize()


loss = nn.CrossEntropyLoss()

model.loss_initialize(loss)



use_apex = False
if config['fp16']:
    opt_level = "O1"
    use_apex = True
else:
    opt_level = "O0"
model.manager(use_apex=use_apex, opt_level=opt_level)




if config['resume']:
    model.resume(save_path=save_path, filename='ckpt_%d.t7' % 60)
methods = ['accuracy']
# methods = ['dice', 'precision', 'recall']
# if not os.path.exists(args.comment):
#     os.makedirs(args.comment)
model.inference(epoch=60, test_loader=test_loader, prob=False, source_file='patient_id.csv',
                target_path=config['outdir'],
                replace_origin='.nii.gz', replace_target='')



