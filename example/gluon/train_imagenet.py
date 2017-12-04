import os
import argparse
import shutil
import time
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data import vision

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
    parser.add_argument()
