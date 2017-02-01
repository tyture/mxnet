# coding: utf-8
# pylint: disable=no-member, too-many-lines, redefined-builtin, protected-access, unused-import, invalid-name
# pylint: disable=too-many-arguments, too-many-locals, no-name-in-module, too-many-branches, too-many-statements
"""Image IO API of mxnet."""
from __future__ import absolute_import, print_function
from .base import numeric_types

import os
import random
import logging
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import ndarray as nd
from . import _ndarray_internal as _internal
from ._ndarray_internal import _cvimresize as imresize
from ._ndarray_internal import _cvcopyMakeBorder as copyMakeBorder
from . import io
from . import recordio


def imdecode(buf, **kwargs):
    """Decode an image from string. Requires OpenCV to work.

    Parameters
    ----------
    buf : str/bytes, or numpy.ndarray
        Binary image data.
    flag : int
        0 for grayscale. 1 for colored.
    to_rgb : int
        0 for BGR format (OpenCV default). 1 for RGB format (MXNet default).
    out : NDArray
        Output buffer. Use None for automatic allocation.
    """
    if not isinstance(buf, nd.NDArray):
        buf = nd.array(np.frombuffer(buf, dtype=np.uint8), dtype=np.uint8)
    return _internal._cvimdecode(buf, **kwargs)

def scale_down(src_size, size):
    """Scale down crop size if it's bigger than image size"""
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w*sh)/h, sh
    if sw < w:
        w, h = sw, float(h*sw)/w
    return int(w), int(h)

def resize_short(src, size, interp=2):
    """Resize shorter edge to size"""
    h, w, _ = src.shape
    if h > w:
        new_h, new_w = size*h/w, size
    else:
        new_h, new_w = size, size*w/h
    return imresize(src, new_w, new_h, interp=interp)

def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size"""
    out = nd.crop(src, begin=(y0, x0, 0), end=(y0+h, x0+w, int(src.shape[2])))
    if size is not None and (w, h) != size:
        out = imresize(out, *size, interp=interp)
    return out

def random_crop(src, size, interp=2):
    """Randomly crop src with size. Upsample result if src is smaller than size"""
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def center_crop(src, size, interp=2):
    """Randomly crop src with size. Upsample result if src is smaller than size"""
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = (w - new_w)/2
    y0 = (h - new_h)/2

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def color_normalize(src, mean, std=None):
    """Normalize src with mean and std"""
    src -= mean
    if std is not None:
        src /= std
    return src

def random_size_crop(src, size, min_area, ratio, interp=2):
    """Randomly crop src with size. Randomize area and aspect ratio"""
    h, w, _ = src.shape
    new_ratio = random.uniform(*ratio)
    if new_ratio * h > w:
        max_area = w*int(w/new_ratio)
    else:
        max_area = h*int(h*new_ratio)

    min_area *= h*w
    if max_area < min_area:
        return random_crop(src, size, interp)
    new_area = random.uniform(min_area, max_area)
    new_w = int(np.sqrt(new_area*new_ratio))
    new_h = int(np.sqrt(new_area/new_ratio))

    assert new_w <= w and new_h <= h
    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def fixed_pad(src, x0, y0, w, h, pad_value, size=None, interp=2):
    """Pad src with target dimension, and (optionally) resize it to size"""
    assert x0 <= 0 and y0 <= 0 and w >= src.shape[1] and h >= src.shape[0]
    dst = nd.full((h, w), pad_value)
    dst[:, -y0:src.shape[0]-y0, -x0:src.shape[1]-x0] = src
    if size is not None and (w, h) != size:
        dst = imresize(dst, *size, interp=interp)
    return dst

def ResizeAug(size, interp=2):
    """Make resize shorter edge to size augumenter"""
    def aug(src):
        """Augumenter body"""
        return [resize_short(src, size, interp)]
    return aug

def RandomCropAug(size, interp=2):
    """Make random crop augumenter"""
    def aug(src):
        """Augumenter body"""
        return [random_crop(src, size, interp)[0]]
    return aug

def RandomSizedCropAug(size, min_area, ratio, interp=2):
    """Make random crop with random resizing and random aspect ratio jitter augumenter"""
    def aug(src):
        """Augumenter body"""
        return [random_size_crop(src, size, min_area, ratio, interp)[0]]
    return aug

def CenterCropAug(size, interp=2):
    """Make center crop augmenter"""
    def aug(src):
        """Augumenter body"""
        return [center_crop(src, size, interp)[0]]
    return aug

def RandomOrderAug(ts):
    """Apply list of augmenters in random order"""
    def aug(src):
        """Augumenter body"""
        src = [src]
        random.shuffle(ts)
        for t in ts:
            src = [j for i in src for j in t(i)]
        return src
    return aug

def ColorJitterAug(brightness, contrast, saturation):
    """Apply random brightness, contrast and saturation jitter in random order"""
    ts = []
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    if brightness > 0:
        def baug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-brightness, brightness)
            src *= alpha
            return [src]
        ts.append(baug)

    if contrast > 0:
        def caug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-contrast, contrast)
            gray = src*coef
            gray = (3.0*(1.0-alpha)/gray.size)*nd.sum(gray)
            src *= alpha
            src += gray
            return [src]
        ts.append(caug)

    if saturation > 0:
        def saug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-saturation, saturation)
            gray = src*coef
            gray = nd.sum(gray, axis=2, keepdims=True)
            gray *= (1.0-alpha)
            src *= alpha
            src += gray
            return [src]
        ts.append(saug)
    return RandomOrderAug(ts)

def LightingAug(alphastd, eigval, eigvec):
    """Add PCA based noise"""
    def aug(src):
        """Augumenter body"""
        alpha = np.random.normal(0, alphastd, size=(3,))
        rgb = np.dot(eigvec*alpha, eigval)
        src += nd.array(rgb)
        return [src]
    return aug

def ColorNormalizeAug(mean, std):
    """Mean and std normalization"""
    mean = nd.array(mean)
    std = nd.array(std)
    def aug(src):
        """Augumenter body"""
        return [color_normalize(src, mean, std)]
    return aug

def HorizontalFlipAug(p):
    """Random horizontal flipping"""
    def aug(src):
        """Augumenter body"""
        if random.random() < p:
            src = nd.flip(src, axis=1)
        return [src]
    return aug

def CastAug():
    """Cast to float32"""
    def aug(src):
        """Augumenter body"""
        src = src.astype(np.float32)
        return [src]
    return aug

def ImageDetRandomSelectionAug(ts):
    """Random select one output from augmenter list, return original if fails"""
    assert len(ts) > 0
    def aug(src, label):
        """Augumenter body"""
        random.shuffle(ts)
        # call one at a time to avoid wasting computation
        for t in ts:
            ret = t(src, label)
            if ret:
                random.shuffle(ret)
                return ret[0]
        return [(src, label)]
    return aug

def ImageDetHorizontalFlipAug(p):
    """Image Detection Random horizontal flipping"""
    def aug(src, label):
        """Augmenter body"""
        if random.random() < p:
            src = nd.flip(src, axis=1)
            label[:, [1, 3]] = 1.0 - label[:, [3, 1]]
        return [(src, label)]
    return aug

def ImageDetRandomSizedCropAug(size, min_area, ratio, max_samples=1, max_trials=50,
                               min_overlap=None, max_overlap=None,
                               min_sample_coverage=None, max_sample_coverage=None,
                               min_obj_coverage=None, max_obj_coverage=None,
                               obj_constraint='center', interp=2):
    """Image Detection Random sized Cropping with Constraints"""
    assert 0 < min_area and min_area <= 1
    assert isinstance(ratio, (list, tuple)) and len(ratio) == 2
    assert min(ratio) > 0
    assert 0 < min_overlap and min_overlap <= 1

    def check_crop_constraints(crop_box, label):
        if (not min_overlap and not max_overlap and not min_sample_coverage and
            not max_sample_coverage and not min_obj_coverage and not max_obj_coverage):
            return True
        # batch compute intersections and unions
        intersect_areas = np.maximum(0, np.minimum(label[:, 3], crop_box[2])
            - np.maximum(label[:, 1], crop_box[0]))
            * np.maximum(0, np.minimum(label[:, 4], crop_box[3])
            - np.maximum(label[:, 2], crop_box[1]))
        crop_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
        obj_areas = (label[:, 3] - label[:, 1]) * (label[:, 4] - label[:, 2])
        for i in range(label.shape[0]):
            if min_overlap or max_overlap:
                inter = intersect_areas[i]
                overlap = inter / (crop_area + obj_areas[i] - inter)
                if min_overlap and overlap < min_overlap:
                    continue
                if max_overlap and overlap > max_overlap:
                    continue
            if min_sample_coverage or max_sample_coverage:
                inter = intersect_areas[i]
                coverage = inter / crop_area
                if min_sample_coverage and coverage < min_sample_coverage:
                    continue
                if max_sample_coverage and coverage > max_sample_coverage:
                    continue
            if min_obj_coverage or max_obj_coverage:
                inter = intersect_areas[i]
                coverage = inter / obj_areas[i]
                if min_obj_coverage and coverage < min_obj_coverage:
                    continue
                if max_obj_coverage and coverage > max_obj_coverage:
                    continue
            return True
        return False

    def transform_crop_labels(crop_box, label):
        """Transform label according to crop region """
        if obj_constraint == 'center':
            center_x = 0.5 * (label[:, 1] + label[:, 3])
            center_y = 0.5 * (label[:, 2] + label[:, 4])
            valid_mask = np.where(np.all(
                [center_x > crop_box[0],
                center_x < crop_box[2],
                center_y > crop_box[1],
                center_y < crop_box[3]]))[0]
            label = label[valid_mask, :]
        elif isinstance(obj_constraint, numeric_types):
            min_overlap = float(obj_constraint)
            intersect_areas = np.maximum(0, np.minimum(label[:, 3], crop_box[2])
                - np.maximum(label[:, 1], crop_box[0]))
                * np.maximum(0, np.minimum(label[:, 4], crop_box[3])
                - np.maximum(label[:, 2], crop_box[1]))
            obj_areas = (label[:, 3] - label[:, 1]) * (label[:, 4] - label[:, 2])
            valid_mask = np.where(intersect_areas / obj_areas > min_overlap)[0]
            label = label[valid_mask, :]
        else:
            raise ValueError("Unknown constraint" + str(obj_constraint))

        new_w = crop_box[2] - crop_box[0]
        new_h = crop_box[3] - crop_box[1]
        label[:, 1] = np.maximum(0, (label[:, 1] - crop_box[0]) / new_w)
        label[:, 2] = np.maximum(0, (label[:, 2] - crop_box[1]) / new_h)
        label[:, 3] = np.minimum(1, (label[:, 3] - crop_box[0]) / new_w)
        label[:, 4] = np.minimum(1, (label[:, 4] - crop_box[1]) / new_h)
        return label

    def aug(src, label):
        """Augmenter body"""
        h, w, _ = src.shape
        hh, ww = (float(h), float(w))
        ret = []
        for t in range(max_trials):
            new_ratio = random.uniform(*ratio)
            if new_ratio * h > w:
                max_area = w * int(w / new_ratio)
            else:
                max_area = h * int(h * new_ratio)
            new_area = min_area * h * w
            if max_area < new_area:
                 new_w, new_h = scale_down((w, h), size)
                 x0 = random.randint(0, w - new_w)
                 y0 = random.randint(0, h - new_h)
            else:
                new_area = random.uniform(new_area, max_area)
                new_w = int(np.sqrt(new_area*new_ratio))
                new_h = int(np.sqrt(new_area/new_ratio))
                assert new_w <= w and new_h <= h
                x0 = random.randint(0, w - new_w)
                y0 = random.randint(0, h - new_h)
            crop_box = (x0 / ww, y0 / hh, (x0 + new_w)/ww/2, (y0 + new_h)/hh/2)
            if check_crop_constraints(crop_box, label):
                label = transform_crop_labels(crop_box, label)
                src = fixed_crop(src, x0, y0, new_w, new_h, size=size, interp=interp)
                ret.append((src, label))
            if len(ret) >= max_samples:
                break
        return ret
    return aug

def ImageDetRandomSizedPadAug(p, size, max_area, padval=128.0, interp=2):
    """Image Detection Random Sized Padding"""
    assert max_area > 1

    def transform_pad_labels(pad_box, label):
        new_w = pad_box[2] - pad_box[0]
        new_h = pad_box[3] - pad_box[1]
        label[:, 1] = np.maximum(0, (label[:, 1] - pad_box[0]) / new_w)
        label[:, 2] = np.maximum(0, (label[:, 2] - pad_box[1]) / new_h)
        label[:, 3] = np.minimum(1, (label[:, 3] - pad_box[0]) / new_w)
        label[:, 4] = np.minimum(1, (label[:, 4] - pad_box[1]) / new_h)
        return label

    def aug(src, label):
        """Augmenter body"""
        if random.random() < p:
            h, w, _ = src.shape
            hh, ww = (float(h), float(w))
            expand_ratio = random.uniform((1.05, max_area))
            new_w = expand_ratio * w
            new_h = expand_ratio * h
            off_w = random.uniform((0, new_w - w))
            off_h = random.uniform((0, new_h - h))
            pad_box = (-off_w / ww, -off_h / hh, (new_w - off_w) / ww, (new_h - off_h) / hh)
            label = transform_pad_labels(pad_box, label)
            src = fixed_pad(src, off_w, off_h, new_w, new_h, padval, size=size, interp=interp)
        return [(src, label)]
    return aug

def CreateAugmenter(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False,
                    mean=None, std=None, brightness=0, contrast=0, saturation=0,
                    pca_noise=0, inter_method=2):
    """Create augumenter list"""
    auglist = []

    if resize > 0:
        auglist.append(ResizeAug(resize, inter_method))

    crop_size = (data_shape[2], data_shape[1])
    if rand_resize:
        assert rand_crop
        auglist.append(RandomSizedCropAug(crop_size, 0.3, (3.0/4.0, 4.0/3.0), inter_method))
    elif rand_crop:
        auglist.append(RandomCropAug(crop_size, inter_method))
    else:
        auglist.append(CenterCropAug(crop_size, inter_method))

    if rand_mirror:
        auglist.append(HorizontalFlipAug(0.5))

    auglist.append(CastAug())

    if brightness or contrast or saturation:
        auglist.append(ColorJitterAug(brightness, contrast, saturation))

    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        auglist.append(LightingAug(pca_noise, eigval, eigvec))

    if mean is True:
        mean = np.array([123.68, 116.28, 103.53])
    if std is True:
        std = np.array([58.395, 57.12, 57.375])
    if mean is not None:
        assert std is not None
        auglist.append(ColorNormalizeAug(mean, std))

    return auglist


class ImageIter(io.DataIter):
    """Image data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.

    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.

    To load from raw image files, specify path_imglist and path_root.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch
    data_shape : tuple
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    label_width : int
        dimension of label
    path_imgrec : str
        path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec
    path_imglist : str
        path to image list (.lst)
        Created with tools/im2rec.py or with custom script.
        Format: index\t[one or more label separated by \t]\trelative_path_from_root
    imglist: list
        a list of image with the label(s)
        each item is a list [imagelabel: float or list of float, imgpath]
    path_root : str
        Root folder of image files
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
        Can be slow for HDD.
    part_index : int
        Partition index
    num_parts : int
        Total number of partitions.
    kwargs : ...
        More arguments for creating augumenter. See mx.image.CreateAugmenter
    """
    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, **kwargs):
        super(ImageIter, self).__init__()
        assert(path_imgrec or path_imglist or (isinstance(imglist, list)))
        if path_imgrec:
            print('loading recordio...')
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r')
                self.imgidx = None
        else:
            self.imgrec = None

        if path_imglist:
            print('loading image list...')
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = nd.array([float(i) for i in line[1:-1]])
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            print('loading image list...')
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index)
                index += 1
                if isinstance(img[0], numeric_types):
                    label = nd.array([img[0]])
                else:
                    label = nd.array(img[0])
                result[key] = (label, img[1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root

        assert len(data_shape) == 3 and data_shape[0] == 3
        self.provide_data = [('data', (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [('softmax_label', (batch_size, label_width))]
        else:
            self.provide_label = [('softmax_label', (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width

        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = None

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N/num_parts
            self.seq = self.seq[part_index*C:(part_index+1)*C]
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def next_sample(self):
        """helper function for reading in next sample"""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                if self.imgrec is None:
                    with open(os.path.join(self.path_root, fname), 'rb') as fin:
                        img = fin.read()
                return label, img
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img

    def next(self):
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = [imdecode(s)]
                if len(data[0].shape) == 0:
                    logging.debug('Invalid image, skipping.')
                    continue
                for aug in self.auglist:
                    data = [ret for src in data for ret in aug(src)]
                for d in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = nd.transpose(d, axes=(2, 0, 1))
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size-1-i)

def CreateDetAugmenter(data_shape, resize=0, rand_crop=[], rand_pad=None,
                       rand_mirror=False, mean=None, std=None, brightness=0,
                       contrast=0, saturation=0, pca_noise=0, inter_method=2):
    """Create augmenter lists for image detection spatial transformations"""
    st_auglist = []  # for spatial transforms

    if resize > 0:
        auglist.append(ResizeAug(resize, inter_method))

    if rand_crop:
        crop_augs = []
        for r in rand_crop:
            assert isinstance(r, dict)
            crop_augs.append(ImageDetRandomSizedCropAug(**r))
        st_auglist.append(ImageDetRandomSelectionAug(crop_augs))

    if rand_pad:
        assert isinstance(rand_pad, dict)
        st_auglist.append(ImageDetRandomSizedPadAug(**rand_pad))

    if rand_mirror:
        st_auglist.append(ImageDetHorizontalFlipAug(p))

    auglist = CreateAugmenter(
        data_shape, resize=resize, mean=mean, std=std,
        brightness=brightness, contrast=contrast, saturation=saturation,
        pca_noise=pca_noise, inter_method=inter_method)

    return [st_auglist, auglist]


class ImageDetIter(io.DataIter):
    """Image detection data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.

    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.

    To load from raw image files, specify path_imglist and path_root.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch
    data_shape : tuple
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    path_imgrec : str
        path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec
    path_imglist : str
        path to image list (.lst)
        Created with tools/im2rec.py or with custom script.
        Format: index\t[one or more label separated by \t]\trelative_path_from_root
    imglist: list
        a list of image with the label(s)
        each item is a list [imagelabel: float or list of float, imgpath]
    path_root : str
        Root folder of image files
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
        Can be slow for HDD.
    part_index : int
        Partition index
    num_parts : int
        Total number of partitions.
    kwargs : ...
        More arguments for creating augumenter. See mx.image.CreateDetAugmenter
    """
    def __init__(self, batch_size, data_shape, path_imgrec=None,
                 path_imglist=None, label_shape=None, label_name='label',
                 cache_label=True, label_padval=-1.0, aug_list=None,
                 **kwargs):
        super(ImageDetIter, self).__init__()
        assert(path_imgrec or path_imglist)
        if path_imgrec:
            logging.info('loading recordio...')
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r')
                self.imgidx = None
        else:
            self.imgrec = None

        if path_imglist:
            logging.info('loading image list...')
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = line[1:-1]
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        else:
            self.imglist = None
        self.path_root = path_root

        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = None

        self.label_padval = label_padval
        self.label_name = label_name
        self.label_shape = label_shape
        self.reshape(data_shape, batch_size, label_shape)

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N/num_parts
            self.seq = self.seq[part_index*C:(part_index+1)*C]
        if aug_list is None:
            self.auglist = CreateDetAugmenter(data_shape, **kwargs)
        else:
            self.auglist = auglist
        self.cur = 0
        self.reset()

    def reshape(self, data_shape=None, batch_size=None, label_shape=None):
        if batch_size:
            self.batch_size = int(batch_size)
        assert self.batch_size > 0
        if data_shape:
            self.data_shape = data_shape
        assert len(self.data_shape) == 3 and self.data_shape[0] == 3
        if label_shape is None and self.label_shape is None:
            logging.info('estimating proper shape for label...')
            self.reset()
            max_len = 0
            for orig_label in iter(self.next_label):
                max_len = max(max_len, len(orig_label))
            assert len(max_len) > 0 and max_len % 5 == 0
            self.label_shape = (max_len // 5, 5)

        if label_shape:
            if self.label_shape:
                if any(map(a < b for a, b in zip(label_shape, self.label_shape))):
                    logging.warning('label_shape shrinks in certain dims')
            self.label_shape = label_shape
            assert all(v > 0 for v in self.label_shape)
        self.provide_data = [('data', (self.batch_size,) + self.data_shape)]
        self.provide_label = [(self.label_name, (self.batch_size,) + self.label_shape)]

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def next_label(self):
        """helper function trying to retrieve labels in next sample"""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label
                else:
                    return self.imglist[idx][0]
            else:
                label, _ = self.imglist[idx]
                return label
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, _ = recordio.unpack(s)
            return header.label

    def next_sample(self):
        """helper function for reading in next sample"""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                if self.imgrec is None:
                    with open(os.path.join(self.path_root, fname), 'rb') as fin:
                        img = fin.read()
                return label, img
        else:
            s = self.imgrec.read()
            self.cur += 1
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img

    def next(self):
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.full(self.provide_label[0][1], self.label_padval)
        i = 0
        try:
            while i < batch_size:
                label, s = self.next_sample()
                label = [np.array(label).reshape((-1, 5))]
                data = [imdecode(s)]
                if len(data[0].shape) == 0:
                    logging.debug('Invalid image, skipping.')
                    continue
                # apply sptial transformations before other augs
                for aug in self.auglist[0]:
                    pairs = [ret for src, l in zip(data, label) for ret in aug(src, l)]
                for aug in self.auglist[1]:
                    pairs = [(ret, l) for src, l in pairs for ret in aug(src)]
                for d, l in pairs:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = nd.transpose(d, axes=(2, 0, 1))
                    slices = [slice(0, i) for i in l.shape]
                    batch_label[i][slices] = l
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size-1-i)
