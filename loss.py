import torch
import torch.nn.functional as F
import math
import copy

class NoisyLabelLoss(torch.nn.Module):
    def __init__(self, args, num_classes=10):
        super(NoisyLabelLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.A = args.A

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        min_clamp = math.exp(self.A)
        label_one_hot = torch.clamp(label_one_hot, min=min_clamp, max=1.0)
        rce = torch.mean((-1*torch.sum(pred * torch.log(label_one_hot), dim=1)))

        loss = ce * self.alpha + rce * self.beta
        return loss

class SmallLoss(torch.nn.Module):
    def __init__(self, args, num_classes=10):
        super(SmallLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.A = args.A

    def forward(self, pred, labels, clean_rate=1):
        # ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_ce = copy.deepcopy(label_one_hot)
        min_clamp = math.exp(self.A)
        label_one_hot = torch.clamp(label_one_hot, min=min_clamp, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        ce  = (-1 * torch.sum(label_ce * torch.log(pred), dim=1))
        # print('ce size = ',ce.size())
        rank = F.softmax(rce, dim=0) * self.beta + F.softmax(ce, dim=0) * self.alpha
        _, indices = rank.sort()

        # num_clean = int(pred.size(0) * (1-self.args.eta))
        num_clean = int(pred.size(0) * clean_rate)
        indices = indices[:num_clean]
        # print(ce[indices].shape)
        loss = ce[indices].mean() * self.alpha + rce[indices].mean() * self.beta
        return loss
