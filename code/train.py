from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from network import ResNet18
from dataset import Dataset

parser = argparse.ArgumentParser(description='Kaggle: Humpback Whale Identification')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='ResNet18', type=str, metavar='Model',
                    help='model type: ResNet18, ResNet34, ResNet50, ResNet101')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='/data2/shentao/DATA/Kaggle/Whale/raw/', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='train.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='test.txt', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='../models/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=5004, type=int,
                    metavar='N', help='number of classes (default: 99891)')

def main():
    global args
    args = parser.parse_args()

    # create Light CNN for face recognition
    if args.model == 'ResNet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=args.num_classes)
    elif args.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=args.num_classes)
    elif args.model == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        model = ResNet50(backbone, num_classes=args.num_classes)
    elif args.model == 'ResNet101':
        backbone = models.resnet101(pretrained=True)
        model = ResNet101(backbone, num_classes=args.num_classes)
    else:
        print('Error model type\n')

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print(model)

    # large lr for last fc parameters
    params = []
    for name, value in model.named_parameters():
        if 'bias' in name:
            if 'fc' in name:
                params += [{'params':value, 'lr': 20 * args.lr, 'weight_decay': 0}]
            else:
                params += [{'params':value, 'lr': 2 * args.lr, 'weight_decay': 0}]
        else:
            if 'fc' in name:
                params += [{'params':value, 'lr': 10 * args.lr}]
            else:
                params += [{'params':value, 'lr': 1 * args.lr}]

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #load image
    train_loader = torch.utils.data.DataLoader(
        Dataset(root=args.root_path, fileList=args.train_list, 
            transform=transforms.Compose([ 
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
            ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Dataset(root=args.root_path, fileList=args.val_list, 
            transform=transforms.Compose([ 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)   

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        criterion.cuda()

    validate(val_loader, model, criterion)    

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        save_name = args.save_path + args.model +'_' + str(epoch+1) + '.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'prec1': prec1,
        }, save_name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        #input      = input.cuda()
        #target     = target.cuda()
        input_var  = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        output = model(input_var)
        loss   = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #input      = input.cuda()
        #target     = target.cuda()
        input_var  = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()
        #print(input_var, target_var)

        # compute output
        output, _ = model(input_var)
        loss   = criterion(output, target_var)

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


    print('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))

    return top1.avg

def save_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step  = 10
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t().cuda()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
