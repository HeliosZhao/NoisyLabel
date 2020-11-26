import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import random
import os.path as osp
import sys
from torch.backends import cudnn
import time
from datetime import timedelta

from trainer import Trainer, SmallLossTrainer
from dataset import NOISE_CIFAR10
from model import Net
from utils import save_checkpoint, Logger
from loss import NoisyLabelLoss, SmallLoss


def get_data(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = NOISE_CIFAR10(
        root=args.root, train=True, noise_rate=args.eta, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = NOISE_CIFAR10(
        root=args.root, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return trainloader, testloader

def main():
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     cudnn.deterministic = True

    main_worker(args)

def main_worker(args):

    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    start_epoch = 0
    best_accuracy = 0

    model = Net()
    model = model.cuda()
    model = nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train_loader, test_loader = get_data(args)

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        # args.start_epoch = 0
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    if args.loss == 'sl':
        criterion = NoisyLabelLoss(args)
        trainer = Trainer(args, model, criterion)
    elif args.loss == 'small':
        criterion = SmallLoss(args)
        trainer = SmallLossTrainer(args, model, criterion)
    else:
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(args, model, criterion)

    if args.evaluate:
        accuracy = trainer.inference(test_loader)
        print(' Evaluate Accuracy : {:5.2%}'.format(accuracy))
        return

    for epoch in range(start_epoch, args.epochs):
        print('==> start training epoch {}'.format(epoch))
        torch.cuda.empty_cache()
        print(' learning rate = ', optimizer.param_groups[0]['lr'])
        trainer.train(train_loader, optimizer, epoch)

        lr_scheduler.step(epoch+1)

        if (epoch+1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            accuracy = trainer.inference(test_loader)
            is_best = (accuracy > best_accuracy)
            best_accuracy = max(accuracy, best_accuracy)
            ckpt_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_accuracy': best_accuracy,
                'optim': optimizer.state_dict(),
            }

            save_checkpoint(ckpt_dict, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  accuracy: {:5.2%}  best: {:5.2%}{}\n'.
                  format(epoch, accuracy, best_accuracy, ' *' if is_best else ''))

    print('==> Test with the best model:')
    checkpoint = torch.load(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    accuracy = trainer.inference(test_loader)
    print('The best model ----> Epoch {:3d} \t best accuracy : {:5.2%}'.format(checkpoint['epoch'], accuracy))
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-l', '--loss', type=str, default='sl')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=120)

    # training configs
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--eta', type=float, default=0.6,
                        help="noise rate")

    # loss
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--A', type=int, default=-4)
    parser.add_argument('--T', type=float, default=10.)
    # parser.add_argument('--CES', action='store_true',
    #                     help="use CES loss")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--root', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")

    main()