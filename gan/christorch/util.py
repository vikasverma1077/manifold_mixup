from __future__ import print_function

import torch
import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

"""
Code borrowed from Pedro Costa's vess2ret repo:
https://github.com/costapt/vess2ret
"""
def convert_to_rgb(img, is_grayscale=False):
    """Given an image, make sure it has 3 channels and that it is between 0 and 1."""
    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))
    img_ch, _, _ = img.shape
    if img_ch != 3 and img_ch != 1:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))
    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)
    if not is_grayscale:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.
    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)

def rnd_crop(img, data_format='channels_last'):
    assert data_format in ['channels_first', 'channels_last']
    from skimage.transform import resize
    if data_format == 'channels_last':
        # (h, w, f)
        h, w = img.shape[0], img.shape[1]
    else:
        # (f, h, w)
        h, w = img.shape[1], img.shape[2]
    new_h, new_w = int(0.1*h + h), int(0.1*w + w)
    # resize only works in the format (h, w, f)
    if data_format == 'channels_first':
        img = img.swapaxes(0,1).swapaxes(1,2)
    # resize
    img_upsized = resize(img, (new_h, new_w))
    # if channels first, swap back
    if data_format == 'channels_first':
        img_upsized = img_upsized.swapaxes(2,1).swapaxes(1,0)
    h_offset = np.random.randint(0, new_h-h)
    w_offset = np.random.randint(0, new_w-w)
    if data_format == 'channels_last':
        final_img = img_upsized[h_offset:h_offset+h, w_offset:w_offset+w, :]
    else:
        final_img = img_upsized[:, h_offset:h_offset+h, w_offset:w_offset+w]
    return final_img

def min_max_then_tanh(img):
    img2 = np.copy(img)
    # old technique: if image is in [0,255],
    # if grayscale then divide by 255 (putting it in [0,1]), or
    # if colour then subtract 127.5 and divide by 127.5, putting it in [0,1].
    # we do: (x - 0) / (255)
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    img2 = (img2 - 0.5) / 0.5
    return img2

def min_max(img):
    img2 = np.copy(img)
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    return img2

def zmuv(img):
    img2 = np.copy(img)
    for i in range(0, img2.shape[0]):
        img2[i, ...] = (img2[i, ...] - np.mean(img2[i, ...])) / np.std(img2[i,...]) # zmuv
    #print np.min(img2), np.max(img2)
    return img2

def swap_axes(img):
    img2 = np.copy(img)
    img2 = img2.swapaxes(3,2).swapaxes(2,1)
    return img2

def int_to_ord(labels, num_classes):
    """
    Convert integer label to ordinal label.
    """
    ords = np.ones((len(labels), num_classes-1))
    for i in range(len(labels)):
        if labels[i]==0:
            continue
        ords[i][0:labels[i]] *= 0.
    return ords

def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

def get_gpu_mem_used():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        totalMemory = nvmlDeviceGetMemoryInfo(handle)
        return totalMemory.used
    except Exception:
        return -1

####################################################################

def test_image_folder(batch_size):
    import torchvision.transforms as transforms
    # loads images in [0,1] initially
    loader = ImageFolder(root="/data/lisa/data/beckhamc/dr-data/train_sample",
                         transform=transforms.Compose([
                             transforms.Scale(256),
                             transforms.CenterCrop(256),
                             transforms.ToTensor(),
                             transforms.Lambda(lambda img: (img-0.5)/0.5)
                         ])
    )
    train_loader = DataLoader(
        loader, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader

import torch.utils.data.dataset as dataset

class NumpyDataset(dataset.Dataset):
    def __init__(self, X, ys, preprocessor=None, rnd_state=np.random.RandomState(0), reorder_channels=False):
        self.X = X
        if ys is not None:
            # => we're dealing with classifier iterator
            if type(ys) != list:
                ys = [ys]
            for y in ys:
                assert len(y) == len(X)
        self.ys = ys
        self.N = len(X)
        #self.keras_imgen = keras_imgen
        self.preprocessor = preprocessor
        self.rnd_state = rnd_state
        self.reorder_channels = reorder_channels
    def __getitem__(self, index):
        xx = self.X[index]
        if self.ys != None:
            yy = []
            for y in self.ys:
                yy.append(y[index])
            yy = np.asarray(yy)
        if self.preprocessor is not None:
            seed = self.rnd_state.randint(0, 100000)
            #xx = self.keras_imgen.flow(xx[np.newaxis], None, batch_size=1, seed=seed, shuffle=False).next()[0]
            xx = self.preprocessor(xx)
        if self.reorder_channels:
            xx = xx.swapaxes(2,1).swapaxes(1,0)
        if self.ys is not None:
            return xx, yy
        else:
            return xx
    def __len__(self):
        return self.N

from PIL import Image

class DatasetFromFolder(Dataset):
    """
    Specify specific folders to load images from.

    Notes
    -----
    Courtesy of:
    https://github.com/togheppi/CycleGAN/blob/master/dataset.py
    With some extra modifications done by me.
    """
    def __init__(self, image_dir, images=None, transform=None,
                 append_label=None, bit16=False):
        """
        Parameters
        ----------
        image_dir: directory where the images are located
        images: a list of images you want instead. If set to `None` it gets all
          images in the directory specified by `image_dir`.
        transform:
        fliplr: enable left/right flip augmentation?
        append_label: if an int is provided, then `__getitem__` will return
          not just the image x, but (x,y), where y denotes the label. This
          means that this iterator could also be used for classifiers.
        """
        super(DatasetFromFolder, self).__init__()
        self.input_path = image_dir
        if images is None:
            self.image_filenames = [x for x in
                                    sorted(os.listdir(self.input_path))]
        else:
            if type(images) != set:
                images = set(images)
            self.image_filenames = [os.path.join(image_dir, fname)
                                    for fname in images]
        self.transform = transform
        self.append_label = append_label
        self.bit16 = bit16
    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path,
                              self.image_filenames[index])
        if self.bit16:
            from skimage.io import imread
            img = imread(img_fn)
            img = img.astype("float32") / 65535.
            img = (img*255.).astype("uint8")
            img = Image.fromarray(img)
        else:
            img = Image.open(img_fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.append_label is not None:
            yy = np.asarray([self.append_label])
            return img, yy
        else:
            return img
    def __len__(self):
        return len(self.image_filenames)

class ImagePool():
    """
    Used to implement a replay buffer for CycleGAN.

    Notes
    -----
    Original code:
    https://github.com/togheppi/CycleGAN/blob/master/utils.py
    Unlike the original implementation, the buffer's images
      are stored on the CPU, not the GPU. I am not sure whether
      this is really worth the effort -- you'd be doing a bit of
      back and forth copying to/fro the GPU which could really
      slow down the training loop I suspect.
    """
    def __init__(self, pool_size):
        """
        use_cuda: if `True`, store the buffer on GPU. This is
          not recommended for large models!!!
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = [] # stored on cpu, NOT gpu

    def query(self, images):
        from torch.autograd import Variable
        if self.pool_size == 0:
            return images.detach()
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image.cpu())
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image.cpu()
                    # since tmp is on cpu, cuda it when
                    # we append it to return images
                    return_images.append(tmp.cuda())
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return Variable(return_images)
