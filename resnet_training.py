# -*- coding: utf-8 -*-
"""
**Resnet** from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchvision.transforms as transforms

import os
import time
import sys

"""### RESNET50"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# The below code was taken from the appendix of the paper "Barrage of Random Transforms for Adversarially Robust Defense"
# This paper can be found at this link: http://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf.

import numpy as np
import random
from skimage import color, transform, morphology
import skimage
from io import BytesIO
import PIL
from scipy import fftpack

# Helper functions for some transforms
def randUnifC(low, high, params=None):
  p = np.random.uniform()
  if params is not None:
    params.append(p)
  return (high-low)*p + low

def randUnifI(low, high, params=None):
  p = np.random.uniform()
  if params is not None:
    params.append(p)
  return round((high-low)*p + low)

def randLogUniform(low, high, base=np.exp(1)):
  div = np.log(base)
  return base**np.random.uniform(np.log(low)/div, np.log(high)/div)

##### TRANSFORMS BELOW #####
def colorPrecisionReduction(img):
  scales = [np.asscalar(np.random.random_integers(8, 200)) for x in range(3)]
  multi_channel = np.random.choice(2) == 0
  params = [multi_channel] + [s/200.0 for s in scales]
  
  if multi_channel:
    img = np.round(img*scales[0])/scales[0]
  else:
    for i in range(3):
      img[:,:,i] = np.round(img[:,:,i]*scales[i]) / scales[i]

  return img

def jpegNoise(img):
  quality = np.asscalar(np.random.random_integers(55, 95))
  params = [quality/100.0]
  pil_image = PIL.Image.fromarray((img*255.0).astype(np.uint8))
  f = BytesIO()
  pil_image.save(f, format='jpeg', quality=quality)
  jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0
  return jpeg_image

def swirl(img):
  strength = (2.0-0.01)*np.random.random(1)[0] + 0.01
  c_x = np.random.random_integers(1, 256)
  c_y = np.random.random_integers(1, 256)
  radius = np.random.random_integers(10, 200)
  params = [strength/2.0, c_x/256.0, c_y/256.0, radius/200.0]
  img = skimage.transform.swirl(img, rotation=0, strength=strength, radius=radius, center=(c_x, c_y))
  return img

def fftPerturbation(img):
  r, c, _ = img.shape
  #Everyone gets the same factor to avoid too many weird artifacts
  point_factor = (1.02-0.98)*np.random.random((r,c)) + 0.98
  randomized_mask = [np.random.choice(2)==0 for x in range(3)]
  keep_fraction = [(0.95-0.0)*np.random.random(1)[0] + 0.0 for x in range(3)]
  params = randomized_mask + keep_fraction
  for i in range(3):
    im_fft = fftpack.fft2(img[:,:,i])
    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft.shape
    if randomized_mask[i]:
      mask = np.ones(im_fft.shape[:2]) > 0
      im_fft[int(r*keep_fraction[i]):int(r*(1-keep_fraction[i]))] = 0
      im_fft[:, int(c*keep_fraction[i]):int(c*(1-keep_fraction[i]))] = 0
      mask = ~mask
      #Now things to keep = 0, things to remove = 1
      mask = mask * ~(np.random.uniform(size=im_fft.shape[:2] ) < keep_fraction[i])
      #Now switch back
      mask = ~mask
      im_fft = np.multiply(im_fft, mask)
    else:
      im_fft[int(r*keep_fraction[i]):int(r*(1-keep_fraction[i]))] = 0
      im_fft[:, int(c*keep_fraction[i]):int(c*(1-keep_fraction[i]))] = 0
      #Now, lets perturb all the rest of the non-zero values by a relative factor
      im_fft = np.multiply(im_fft, point_factor)
      im_new = fftpack.ifft2(im_fft).real
      #FFT inverse may no longer produce exact same range, so clip it back
      im_new = np.clip(im_new, 0, 1)
      img[:,:,i] = im_new
  return img

## Color Space Group Below
def alterHSV(img):
  img = color.rgb2hsv(img)
  params = []
  #Hue
  img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
  #Saturation
  img[:,:,1] += randUnifC(-0.25, 0.25, params=params)
  #Value
  img[:,:,2] += randUnifC(-0.25, 0.25, params=params)
  img = np.clip(img, 0, 1.0)
  img = color.hsv2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

def alterXYZ(img):
  img = color.rgb2xyz(img)
  params = []
  #X
  img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
  #Y
  img[:,:,1] += randUnifC(-0.05, 0.05, params=params)
  #Z
  img[:,:,2] += randUnifC(-0.05, 0.05, params=params)
  img = np.clip(img, 0, 1.0)
  img = color.xyz2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

def alterLAB(img):
  img = color.rgb2lab(img)
  params = []
  #L
  img[:,:,0] += randUnifC(-5.0, 5.0, params=params)
  #a
  img[:,:,1] += randUnifC(-2.0, 2.0, params=params)
  #b
  img[:,:,2] += randUnifC(-2.0, 2.0, params=params)
  # L 2 [0,100] so clip it; a & b channels can have,! negative values.
  img[:,:,0] = np.clip(img[:,:,0], 0, 100.0)
  img = color.lab2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

def alterYUV(img):
  img = color.rgb2yuv(img)
  params = []
  #Y
  img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
  #U
  img[:,:,1] += randUnifC(-0.02, 0.02, params=params)
  #V
  img[:,:,2] += randUnifC(-0.02, 0.02, params=params)
  # U & V channels can have negative values; clip only Y
  img[:,:,0] = np.clip(img[:,:,0], 0, 1.0)
  img = color.yuv2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

## Grey Scale Group Below
def greyScaleMix(img):
  # average of color channels, different contribution for each channel
  ratios = np.random.rand(3)
  ratios /= ratios.sum()
  params = [x for x in ratios]
  img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1] + img[:,:,2] * ratios[2]
  for i in range(3):
    img[:,:,i] = img_g
  return img

def greyScalePartialMix(img):
  ratios = np.random.rand(3)
  ratios/=ratios.sum()
  prop_ratios = np.random.rand(3)
  params = [x for x in ratios] + [x for x in prop_ratios]
  img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1] + img[:,:,2] * ratios[2]
  for i in range(3):
    p = max(prop_ratios[i], 0.2)
    img[:,:,i] = img[:,:,i]*p + img_g*(1.0-p)
  return img

def greyScaleMixTwoThirds(img):
  params = []
  # Pick a channel that will be left alone and remove it from the ones to be averaged
  channels = [0, 1, 2]
  remove_channel = np.random.choice(3)
  channels.remove( remove_channel)
  params.append( remove_channel )
  ratios = np.random.rand(2)
  ratios/=ratios.sum()
  params.append(ratios[0]) #They sum to one, so first item fully specifies the group
  img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
  for i in channels:
    img[:,:,i] = img_g
  return img

def oneChannelPartialGrey(img):
  params = []
  # Pick a channel that will be altered and remove it from the ones to be averaged
  channels = [0, 1, 2]
  to_alter = np.random.choice(3)
  channels.remove(to_alter)
  params.append(to_alter)
  ratios = np.random.rand(2)
  ratios/=ratios.sum()
  params.append(ratios[0]) #They sum to one, so first item fully specifies the group
  img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
  # Lets mix it back in with the original channel
  p = (0.9-0.1)*np.random.random(1)[0] + 0.1
  params.append( p )
  img[:,:,to_alter] = img_g*p + img[:,:,to_alter] *(1.0-p)
  return img

## Denoising Group
def gaussianBlur(img):
  if randUnifC(0, 1) > 0.5:
    sigma = [randUnifC(0.1, 0.8)]*3
  else:
    sigma = [randUnifC(0.1, 0.8), randUnifC(0.1, 0.8), randUnifC(0.1, 0.8)]
    img[:,:,0] = skimage.filters.gaussian(img[:,:,0], sigma=sigma[0])
    img[:,:,1] = skimage.filters.gaussian(img[:,:,1], sigma=sigma[1])
    img[:,:,2] = skimage.filters.gaussian(img[:,:,2], sigma=sigma[2])
  return img

def chambolleDenoising(img):
  params = []
  weight = (0.25-0.05)*np.random.random(1)[0] + 0.05
  params.append( weight )
  multi_channel = np.random.choice(2) == 0
  params.append( multi_channel )
  img = skimage.restoration.denoise_tv_chambolle( img, weight=weight, multichannel=multi_channel)
  return img

def nonlocalMeansDenoising(img):
  h_1 = randUnifC(0, 1)
  params = [h_1]
  sigma_est = np.mean(skimage.restoration.estimate_sigma(img,multichannel=True) )
  h = (1.15-0.6)*sigma_est*h_1 + 0.6*sigma_est
  #If false, it assumes some weird 3D stuff
  multi_channel = np.random.choice(2) == 0
  params.append( multi_channel )
  #Takes too long to run without fast mode.
  fast_mode = True
  patch_size = np.random.random_integers(5, 7)
  params.append(patch_size)
  patch_distance = np.random.random_integers(6, 11)
  params.append(patch_distance)
  if multi_channel:
    img = skimage.restoration.denoise_nl_means( img,h=h, patch_size=patch_size,patch_distance=patch_distance,fast_mode=fast_mode )
  else:
      for i in range(3):
          sigma_est = np.mean(skimage.restoration.estimate_sigma(img[:,:,i], multichannel=True ) )
          h = (1.15-0.6)*sigma_est*params[i] + 0.6*sigma_est
          img[:,:,i] = skimage.restoration.denoise_nl_means(img[:,:,i], h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode )
  return img

def applyTransforms(img):
  img = np.array(img)
  allTransforms = [[colorPrecisionReduction], [jpegNoise], [swirl], [fftPerturbation], [alterHSV, alterXYZ, alterLAB, alterYUV], [greyScaleMix, greyScalePartialMix, greyScaleMixTwoThirds, oneChannelPartialGrey], [gaussianBlur, chambolleDenoising, nonlocalMeansDenoising]]
  numTransforms = random.randint(0, 5)
  
  img = img / 255.0

  for i in range(numTransforms):
      transformGroup = random.choice(allTransforms)
      transform = random.choice(transformGroup)
      
      img = transform(img)

      allTransforms.remove(transformGroup)
    
  return torch.from_numpy(np.swapaxes(img, 0, 2)).float()

# The below code was adapted from https://github.com/kuangliu/pytorch-cifar
'''Train CIFAR10 with PyTorch.'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

# Code for normalization from the Advex-UAR codebase by Daniel Kang, et al.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# The normal, untransformed training data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize])

# BaRT-transformed training data
transform_train_bart = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([applyTransforms], p=1),
    normalize])

# BaRT-transformed testing data
transform_test_bart = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([applyTransforms], p=1),
    normalize])

# The normal, untransformed testing data
transform_test = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# BaRT trainset
transform_trainset = torchvision.datasets.CIFAR10(root='./transform_data', train=True, download=True, transform=transform_train_bart)
transform_trainloader = torch.utils.data.DataLoader(transform_trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# BaRT testset
transform_testset = torchvision.datasets.CIFAR10(root='./transform_data', train=False, download=True, transform=transform_test_bart)
transform_testloader = torch.utils.data.DataLoader(transform_testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet(Bottleneck, [3,4,6,3]).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(epoch):
    if epoch < 90:
        return 1
    elif epoch >= 90 and epoch < 135:
        print("Learning rate of 0.01")
        return 0.1
    else:
        print("Learning rate of 0.001")
        return 0.01

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = adjust_learning_rate)

# The below code was adapted from https://github.com/kuangliu/pytorch-cifar
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('Train Epoch: %d | Batch ID: %d/%d | Train Loss: %.3f | Acc: %.3f%% (%d/%d)' %(epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def transform_train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(transform_trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('Transform Epoch: %d | Batch ID: %d/%d | Train Loss: %.3f | Acc: %.3f%% (%d/%d)' %(epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print('Test Epoch: %d | Batch ID: %d/%d | Test Loss: %.3f | Acc: %.3f%% (%d/%d)' %(epoch, batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    print("Saving every epoch")
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    pathname = './checkpoint/ckpt' + str(epoch) + '.pth'
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, pathname)

def transform_test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(transform_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 20 == 0:
              print('Transform Test Epoch: %d | Batch ID: %d/%d | Test Loss: %.3f | Acc: %.3f%% (%d/%d)' %(epoch, batch_idx, len(transform_testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("Saving every epoch")
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    pathname = './checkpoint/transform-ckpt' + str(epoch) + '.pth'
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, pathname)

# USE THIS TO KEEP TRAINING FROM A PREVIOUS CHECKPOINT
#pth_file_path = "/ckpt3.pth"
#checkpoint = torch.load(pth_file_path)
#net.load_state_dict(checkpoint['net'])

#180 epochs of non-BaRT training
for epoch in range(180):
    print(epoch)
    train(epoch)
    test(epoch)
    scheduler.step()

#20 epochs of training on BaRT-transformed images
for epoch in range(20):
    transform_train(epoch)
    test(epoch)

state = {
    'net': net.state_dict(),
}

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/final_trained_resnet.pth')
