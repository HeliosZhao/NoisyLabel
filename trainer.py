import torch
import time
import torch.nn as nn
from utils import AverageMeter, accuracy
import cv2
import os

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


class CleanNetTrainer(Trainer):
    def train(self, train_loader, optimizer, epoch, train_iters=0):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc_meters = AverageMeter()
        acc5_meters = AverageMeter()

        with torch.autograd.set_detect_anomaly(True):
            for i in range(train_iters):
                batch_data = train_loader.next()
                imgs, labels, true_labels = self._parse_data(batch_data)
                # print(labels)
                # print(labels.shape)
                start_time = time.time()
                pred, clean_indices = self._forward(imgs)

                # self.image_detect(imgs, clean_indices, labels, i)

                labels = self.match_label(labels, clean_indices)
                true_labels = self.match_label(true_labels, clean_indices)
                # print('     labels = ', labels)
                # print('true labels = ', true_labels)
                print('differences = ', true_labels-labels)
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
                          .format(epoch, i + 1, train_iters,
                                  batch_time.val, batch_time.avg,
                                  losses.val, losses.avg,
                                  acc_meters.val, acc5_meters.val))

    def _parse_data(self, inputs):
        imgs, labels, true_labels = inputs
        return imgs.cuda(), labels.cuda(), true_labels.cuda()

    def match_label(self, labels, indices):
        new_labels = labels
        new_labels = new_labels.view(-1, self.args.num_instances)
        # print(new_labels)
        match_label = []
        for i in range(indices.size(0)):
            match_label.append(new_labels[i][indices[i]])
        match_label = torch.cat(match_label, dim=0)
        return match_label

    def image_detect(self,imgs, indices, labels, iter):
        if not os.path.exists('/home/fxyang/zyy/code/noisylabel/logs/save_imgs'):
            os.makedirs('/home/fxyang/zyy/code/noisylabel/logs/save_imgs')
        imgs = imgs.view(-1, self.args.num_instances, 3, 32, 32).permute(0,1,3,4,2)
        labels = labels.view(-1, self.args.num_instances)
        for i in range(imgs.size(0)):
            label = labels[i]
            # print(label)
            for ind in indices[i]:
                if label[ind] == 5:
                    save_im = imgs[i][ind].cpu().numpy() * 255.
                    save_path = 'iter_{}_img_{}_label_{}_ind_{}.png'.format(iter, i, label[ind], ind)
                    cv2.imwrite(os.path.join('/home/fxyang/zyy/code/noisylabel/logs/save_imgs',save_path),save_im)



