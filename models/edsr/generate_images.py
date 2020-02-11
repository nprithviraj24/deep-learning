import torch

import math
import torch.nn as nn
import torch.nn.functional as F

import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.vgg as vgg
from tqdm import tqdm_notebook
import sys
# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter

TB = False
batch_size = 8
augment_Gaussian_blur = True
SAVE_EVERY = 10

if sys.argv[1] and sys.argv[1] == '--tensorboard':
    TB = True
    assert sys.argv[2], "Enter Tensorboard filename" 
    writer = SummaryWriter("Cycle-EDSR/test/"+str(sys.argv[2]))
    print("Saving results in Cycle-EDSR/test/"+str(sys.argv[2]))

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

def imshow(img):
    npimg = img.detach().cpu().numpy()
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(1.0)

def recover_image(img):
    return ((img * np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) + np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)) ).transpose(0, 2, 3, 1) * 255. ).clip(0, 255).astype(np.uint8)


print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(convert_size(torch.cuda.memory_allocated(device=device)))



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




def gaussian_kernel(size, sigma=2, dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.

      kernel_size = 2*size + 1
      kernel_size = [kernel_size] * dim
      sigma = [sigma] * dim
      kernel = 1
      meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

      for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
          mean = (size - 1) / 2
          kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

      # Make sure sum of values in gaussian kernel equals 1.
      kernel = kernel / torch.sum(kernel)
      # Reshape to depthwise convolutional weight
      kernel = kernel.view(1, 1, *kernel.size())
      kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
      return kernel

def _gaussian_blur(x, size, Blur_sigma):
      kernel = gaussian_kernel(size=size, sigma=Blur_sigma)
      kernel_size = 2*size + 1
      x = x[None,...]
      padding = int((kernel_size - 1) / 2)
      x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
      x = torch.squeeze(F.conv2d(x, kernel, groups=3))
      return x



class NoiseAndBlur():
    """Adds gaussian noise to a tensor.

        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     Noise(0.1, 0.05)),
        >>> ])

    """
    def __init__(self, mean, stddev, image_size, applyBlur, Blur_sigma, Blur_ker_size):
        self.mean = mean
        self.stddev = stddev
        self.image_size = image_size
        self.Blur_sigma = Blur_sigma
        self.Blur_ker_size = Blur_ker_size
        self.applyBlur = applyBlur

    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        if self.applyBlur == True:
          return _gaussian_blur(tensor.add_(noise), self.Blur_ker_size, self.Blur_sigma)
        else:
          return tensor.add_(noise)

def get_data_loader(image_type, image_dir='lrtohr', image_size=64, batch_size=batch_size, num_workers=0):
    """Returns training and test data loaders for a given image type
    """

    # resize and normalize the images
    transform1 = transforms.Compose([transforms.Resize((32, 32)) # resize to 128x128
                                    ,transforms.ToTensor()
                                    # ,NoiseAndBlur(0.1, 0.05, image_size = image_size, applyBlur=augment_Gaussian_blur, Blur_sigma=0, Blur_ker_size = 4)
                                    ,transforms.RandomErasing(p=0.2, scale=(0.00002, 0.001), ratio=(0.0001, 0.0006), value=0, inplace=False)
                                    # , tensor_normalizer()
                                    ])
    # get training and test directories
    # resize and normalize the images
    transform2 = transforms.Compose([transforms.Resize((128,128)), # resize to 128x128
                                    transforms.ToTensor()
                                    # , tensor_normalizer()
                                    ])

    transform0 = transforms.Compose([transforms.Resize((32, 32))
                                    ,transforms.ToTensor()
                                    # ,_gaussian_blur()
                                    # ,NoiseAndBlur(0.1, 0.05, image_size = image_size, applyBlur=augment_Gaussian_blur, Blur_sigma=1, Blur_ker_size = 4)
                                    ,transforms.RandomErasing(p=0.5, scale=(0.00002, 0.001), ratio=(0.0001, 0.0006), value=0, inplace=False)
                                     # , tensor_normalizer()
                                    ])



    if image_type == 'lr':
        image_path = './' + image_dir
        train_path = os.path.join(image_path, image_type)
        test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform1)
        test_dataset = datasets.ImageFolder(test_path, transform1)

        # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if image_type == 'hr':
        image_path = './' + image_dir
        train_path = os.path.join(image_path, image_type)
        test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform2)
        test_dataset = datasets.ImageFolder(test_path, transform2)

            # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    if image_type == 'div2k':
        path = image_dir
        dataset = datasets.ImageFolder(path, transform0)
        n = len(dataset)
        # test_dataset = datasets.ImageFolder(test_path, transform2)

            # create and return DataLoaders
        # data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [1600, n-1600])
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, drop_last=True)



    return train_loader, test_loader

dataloader_X, test_iter_X = get_data_loader(image_type='lr')
dataloader_Y, test_iter_Y = get_data_loader(image_type='hr')

div2k, test_div2k = get_data_loader(image_dir='DIV2K/', image_type='div2k')


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()
        conv=default_conv
        act = nn.ReLU(True)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

G_XtoY = EDSR()

from torch.utils import save_image

PATH =""
upsample4x = nn.Upsample(scale_factor=4, mode='bicubic')
batch = 0
G_XtoY.load_state_dict(torch.load(PATH))
for batch_id, (x, _) in tqdm_notebook(enumerate(test_div2k), total=len(test_div2k)):
    batch +=1
    y = next(iter(dataloader_X))
    assert y.shape == x.shape, "y and x should be same shape"
    with torch.no_grad():
       G_XtoY.eval() # set generators to eval mode for sample generation
       fakeY = G_XtoY(x.to(device))
       bicubicX = upsample4x(x.to(device)) 

       for i in range(0, batch_size):
           save_image(fakeY[i], 'FID/SR/'+str(batch)+'_'+str(i)+'.png')
           save_image(bicubicX[i], 'FID/bicubic/'+str(batch)+'_'+str(i)+'.png')
           save_image(y[i], 'FID/groudTruth/'+str(batch)+'_'+str(i)+'.png')
           
