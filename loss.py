import torch
import torch.nn.functional as F
import math

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

# class NoisyLabelLoss(torch.nn.Module):
#     def __init__(self, args, num_classes=10):
#         super(NoisyLabelLoss, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.alpha = args.alpha
#         self.beta = args.beta
#         self.num_classes = num_classes
#         self.cross_entropy = torch.nn.CrossEntropyLoss()
#
#     def forward(self, pred, labels):
#         # CCE
#         ce = self.cross_entropy(pred, labels)
#
#         # RCE
#         pred = F.softmax(pred, dim=1)
#         pred = torch.clamp(pred, min=1e-7, max=1.0)
#         label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
#         label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
#         rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
#
#         # Loss
#         loss = self.alpha * ce + self.beta * rce.mean()
#         return loss
