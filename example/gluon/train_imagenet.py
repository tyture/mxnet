import os
import argparse
import shutil
import time
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.image import SequentialAug, RandomSizedCropAug, ResizeAug, CenterCropAug, HorizontalFlipAug
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Gluon ImageNet12 training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str,
                        help='path to dataset')
    parser.add_argument('--model', required=True, type=str,
                        help='gluon model name, e.g. resnet18_v1')
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--gpus', default='0', type=str,
                        help='gpus to use, multiple gpus supported as "0,1,2,3"')
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
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
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
    parser.add_argument('--lr-factor', default=0.1, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--lr-steps', default='30,60,90', type=str,
                        help='list of learning rate decay epochs as in str')
    args = parser.parse_args()
    return args

def get_model(model, resume, pretrained):
    """Model initialization."""
    net = gluon.model_zoo.vision.get_model(model, pretrained=pretrained, classes=1000)
    if resume:
        net.load_params(resume)
    elif not pretrained:
        if model in ['alexnet']:
            net.intialize(mx.init.Normal())
        else:
            net.initialize(mx.init.Xavier(magnitude=2))
    net.hybridize()
    return net

def train_transform(image, label):
    image, _ = mx.image.random_size_crop(image, (224, 224), 0.08, (3/4., 4/3.))
    image = mx.nd.image.random_horizontal_flip(image)
    image = mx.nd.image.to_tensor(image)
    image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return image, label

def val_transform(image, label):
    image = mx.image.resize_short(image, 256)
    image, _ = mx.image.center_crop(image, (224, 224))
    image = mx.nd.image.to_tensor(image)
    image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return image, label

def get_dataloader(root, batch_size, num_workers):
    """Dataset loader with preprocessing."""
    train_dir = os.path.join(root, 'train')
    train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
    train_data = DataLoader(train_dataset, batch_size, shuffle=True,
                            last_batch='rollover', num_workers=num_workers)
    val_dir = os.path.join(root, 'val')
    val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
    val_data = DataLoader(val_dataset, batch_size, last_batch='keep', num_workers=num_workers)
    return train_data, val_data

def update_learning_rate(lr, trainer, epoch, ratio, steps):
    """Set the learning rate to the initial value decayed by ratio every N epochs."""
    new_lr = lr * (ratio ** int(np.sum(np.array(steps) < epoch)))
    trainer.set_learning_rate(new_lr)
    return trainer

def validate(net, val_data, metrics, ctx):
    """Validation."""
    for m in metrics:
        m.reset()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x, y in zip(data, label):
            z = net(x)
            outputs.append(z)
        for m in metrics:
            m.update(label, outputs)

    msg = ','.join(['%s=%f'%(m.get()) for m in metrics])
    return msg, m[0].get()[1]

def train(net, train_data, val_data, ctx, args):
    """Training"""
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(5)]
    lr_steps = [int(x) for x in args.lr_steps.split(',') if x.strip()]
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum},
                            kvstore = args.kvstore)

    # start training
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        trainer = update_learning_rate(args.lr, trainer, epoch, args.lr_factor, lr_steps)
        for m in metrics:
            m.reset()
        tic = time.time()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            losses = []
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = criterion(z, y)
                    losses.append(L)
                    outputs.append(z)
                autograd.backward(losses)
            batch_size = args.batch_size
            trainer.step(batch_size)
            for m in metrics:
                m.update(label, outputs)
            if args.log_interval and (i + 1) % args.log_interval == 0:
                msg = ','.join(['%s=%f'%(m.get()) for m in metrics])
                logging.info('Epoch[%d] Batch[%d]\tSpeed: %f samples/sec\t%s',
                             epoch, i, batch_size/(time.time()-btic), msg)
            btic = time.time()

        msg = ','.join(['%s=%f'%(m.get()) for m in metrics])
        logging.info('[Epoch %d] Training: %s', epoch, msg)
        logging.info('[Epoch %d] Training time cost: %f', epoch, time.time()-tic)
        msg, top1 = validate(net, val_data, metrics, ctx)
        logging.info('[Epoch %d] Validation: %s', epoch, msg)

        fname = os.path.join(args.prefix, '%s_%d_acc_%.4f.params'%(args.model, epoch, top1))
        net.save_params(fname)
        logging.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)
        if top1 > best_acc:
            best_acc = top1
            net.save_params(os.path.join(args.prefix, '%s_best_acc_%.4f.params'%(args.model, top1)))


if __name__ == '__main__':
    args = parse_args()
    logging.info(args)
    # get the network
    net = get_model(args.model, args.resume, args.pretrained)
    # get the dataset
    train_data, val_data = get_dataloader(args.data, args.batch_size, args.num_workers)
    # set up contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # start training
    train(net, train_data, val_data, ctx, args)
