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

from network import *
from dataset import WhaleDataset
#from utils import *
#from loss import FocalLoss

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
parser.add_argument('--model', default='ResNet34', type=str, metavar='Model',
                    help='model type: ResNet18, ResNet34, ResNet50, ResNet101')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='/data2/shentao/DATA/Kaggle/Whale/raw/', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='../data/train_full.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='../data/val.txt', type=str, metavar='PATH',
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
        branch = ResNet18(backbone, num_classes=args.num_classes)
    elif args.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        branch = ResNet34(backbone, num_classes=args.num_classes)
    elif args.model == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        branch = ResNet50(backbone, num_classes=args.num_classes)
    elif args.model == 'ResNet101':
        backbone = models.resnet101(pretrained=True)
        branch = ResNet101(backbone, num_classes=args.num_classes)
    elif args.model == 'ResNet152':
        backbone = models.resnet152(pretrained=True)
        branch = ResNet152(backbone, num_classes=args.num_classes)
    else:
        print('Error model type\n')

    head = Head(dim = 512*2)
    if args.cuda:
        branch = torch.nn.DataParallel(branch).cuda()
        head = torch.nn.DataParallel(head).cuda()
    print('Siamese Branch:\n{}\n'.format(branch))
    print('Siamese Head:\n{}\n'.format(head))

    # large lr for last fc parameters
    params = []
    for name, value in head.named_parameters():
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
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #load image
    train_loader = torch.utils.data.DataLoader(
        WhaleDataset(root=args.root_path, fileList=args.train_list, mode='train',
            transform=transforms.Compose([ 
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(30),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),
            ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    # val_loader = torch.utils.data.DataLoader(
    #    Dataset(root=args.root_path, fileList=args.val_list, 
    #        transform=transforms.Compose([ 
    #            transforms.CenterCrop(224),
    #            transforms.ToTensor(),
    #        ])),
    #    batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=False)   

    # define loss function and optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(gamma=2)

    if args.cuda:
        criterion.cuda()

    #validate(val_loader, model, criterion)    

    best_prec1 = 0
    #lrs = []
    #losses_ = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        #lr = adjust_learning_rate(optimizer, epoch)
        #lrs.append(lr)
        #draw_curve(lrs, 0.9, 'learning_rate')

        # train for one epoch
        train(train_loader, branch, head, criterion, optimizer, epoch)
        #loss = train(train_loader, model, criterion, optimizer, epoch)
        #losses_.append(loss)
        #draw_curve(losses_, 0.9, 'loss')

        # evaluate on validation set
        #prec1, prec5 = validate(val_loader, model, criterion)
        #if prec1>best_prec1:
        #    best_prec1 = prec1

        #save_name = args.save_path + args.model +'_' + str(epoch+1) + '.pth.tar'
        if epoch>30 and epoch%10==0:
            save_name = args.save_path + 'Siamese_'+args.model +'_' + str(epoch) +'_best_bce.pth'
            save_checkpoint(model, save_name)


def train(train_loader, branch, head, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    branch.eval()
    head.train()

    end = time.time()
    print('start training...')
    for i, (input1, input2, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input1_var  = torch.autograd.Variable(input1).cuda()
        input2_var  = torch.autograd.Variable(input2).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        print('input1:{}\ninput1:{}\ntarget:{}'.format(input1_var, input2_var, target_var))

        # compute output
        fea1 = branch.forward(input1_var)
        fea2 = branch.forward(input2_var)
        output = head.forward(fea1, fea2)
        #print('label: {}\npredict: {}'.format(target_var, output.size()))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1,5))
        losses.update(loss.data[0], input1_var.size(0))
        top1.update(prec1[0], input1_var.size(0))
        top5.update(prec5[0], input1_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{}/{}][{}/{}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) | '
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        #return loss.data[0]

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var  = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()
        #print(input_var, target_var)
        #print(input_var.shape)

        # compute output
        output = model(input_var)
        #print('label: {}\npredict: {}'.format(target_var, output.size()))
        loss   = criterion(output, target_var)

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


    print('\nTest set: Average loss: {}, Accuracy@1: {}, Accuracy@5: {}\n'.format(losses.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg

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
    step  = 20
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

    #return lr


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
