import torch
import time
import torch.nn as nn
from utils import AverageMeter, accuracy

class Trainer(object):
    def __init__(self, args, model, criterion):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.args = args

    def train(self, train_loader, optimizer, epoch, train_iters=0):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc_meters = AverageMeter()
        acc5_meters = AverageMeter()
        with torch.autograd.set_detect_anomaly(True):
            for i, batch_data in enumerate(train_loader):
                imgs, labels = self._parse_data(batch_data)
                start_time = time.time()
                pred = self._forward(imgs)
                loss = self.criterion(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc, acc5 = accuracy(pred, labels, topk=(1, 5))


                losses.update(loss.item())
                acc_meters.update(acc.item())
                acc5_meters.update(acc5.item())

                batch_time.update(time.time() - start_time)

                if (i + 1) % self.args.print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Loss {:.3f}({:.3f})\t'
                          'Accuracy {:.3f}/{:.3f}'
                          .format(epoch, i + 1, len(train_loader),
                                  batch_time.val, batch_time.avg,
                                  losses.val, losses.avg,
                                  acc_meters.val, acc5_meters.val))



    def inference(self, test_loader):
        test_acc_meters = AverageMeter()
        test_acc5_meters = AverageMeter()
        self.model.eval()
        for i, batch_data in enumerate(test_loader):
            imgs, labels = self._parse_data(batch_data)
            with torch.no_grad():
                pred = self._forward(imgs)
                acc, acc5 = accuracy(pred, labels, topk=(1, 5))
                test_acc_meters.update(acc.item())
                test_acc5_meters.update(acc5.item())

                if (i + 1) % self.args.print_freq == 0:
                    print('Iters: [{}/{}]\t'
                          'Accuracy {:.3f}/{:.3f}\t'
                          'Accuracy5 {:.3f}/{:.3f}'
                          .format(i + 1, len(test_loader),
                                  test_acc_meters.val, test_acc_meters.avg,
                                  test_acc5_meters.val, test_acc5_meters.avg))

        return test_acc_meters.avg

    def _parse_data(self, inputs):
        imgs, labels = inputs
        return imgs.cuda(), labels.cuda()

    def _forward(self, inputs):
        return self.model(inputs)

class SmallLossTrainer(Trainer):
    def train(self, train_loader, optimizer, epoch, train_iters=0):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc_meters = AverageMeter()
        acc5_meters = AverageMeter()
        clean_rate = self.clean_rate(epoch)
        print('clean_rate at epoch {:d} is {:.2f}'.format(epoch, clean_rate))

        with torch.autograd.set_detect_anomaly(True):
            for i, batch_data in enumerate(train_loader):
                imgs, labels = self._parse_data(batch_data)
                start_time = time.time()
                pred = self._forward(imgs)
                loss = self.criterion(pred, labels, clean_rate)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc, acc5 = accuracy(pred, labels, topk=(1, 5))


                losses.update(loss.item())
                acc_meters.update(acc.item())
                acc5_meters.update(acc5.item())

                batch_time.update(time.time() - start_time)

                if (i + 1) % self.args.print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Loss {:.3f}({:.3f})\t'
                          'Accuracy {:.3f}/{:.3f}'
                          .format(epoch, i + 1, len(train_loader),
                                  batch_time.val, batch_time.avg,
                                  losses.val, losses.avg,
                                  acc_meters.val, acc5_meters.val))
    def clean_rate(self, epoch):
        noise_weight = min(1, epoch/self.args.T)
        return (1 - self.args.eta * noise_weight)
