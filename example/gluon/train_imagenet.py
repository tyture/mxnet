import os
import argparse
import shutil
import time
import logging
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Gluon ImageNet12 training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str,
                        help='path to dataset')
    parser.add_argument('--model', required=True, type=str,
                        help='gluon model name, e.g. resnet18_v1')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--epochs', default=120, type=int,
                        help='number of training epochs')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='starting epoch, 0 for fresh training, > 0 to resume')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--log-interval', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--prefix', default='', type=str,
                        help='path to checkpoint')
    parser.add_argument('--resume', default='', type=str,
                        help='path to resuming checkpoint')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--kvstore', default='device', type=str,
                        help='kvstore type')
    args = parser.parse_args()
    return args

def get_model(model, resume, pretrained):
    """Model initialization."""
    net = gluon.data.vision.get_model(model, pretrained)
    if resume:
        net.load_params(resume)
    elif not pretrained:
        if model in ['alexnet']:
            net.intialize(mx.init.Normal())
        else:
            net.initialize(mx.init.Xavier(magnitude=2))

def get_dataloader(root, batch_size, num_workers):
    """Dataset loader with preprocessing."""
    train_dir = os.path.join(root, 'train')
    train_dataset = ImageFolderDataset(
        train_dir, transform=mx.image.RandomSizedCropAug(224, 0.08, (3/4., 4/3.)))
    train_data = DataLoader(train_dataset, batch_size, shuffle=True,
                            last_batch='rollover', num_workers=num_workers)
    val_dir = os.path.join(root, 'val')]
    val_dataset = ImageFolderDataset(val_dir)
    val_data = DataLoader(val_dataset, batch_size, last_batch='keep', num_workers=num_workers)
    return train_data, val_data


class Preprocess(gluon.HybridBlock):
    """Preprocess block for training/testing."""
    def __init__(self, is_train, **kwargs):
        super(Preprocess, self).__init__(**kwargs)
        self.is_train = is_train

    def hybrid_forward(self, F, x, *args):
        if self.is_train:
            x = F.image.random_horizontal_flip(x)
        x = F.image.to_tensor(x)
        x = F.image.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return x

def train(net, epochs, start_epoch, train_data, val_data, ctx):
    """Training"""
    best_acc = 0

if __name__ == '__main__':
    args = parse_args()
    logging.info(args)
    # get the network
