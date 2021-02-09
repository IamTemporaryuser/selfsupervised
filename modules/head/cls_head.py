import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Accuracy
from mmcv.cnn import normal_init

def cross_entropy(pred, label):
    loss = F.cross_entropy(pred, label, reduction='none')
    loss = loss.mean()
    return loss

class LinearClsHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 topk=(1, )):
        super(LinearClsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.accuracy = Accuracy(topk=self.topk)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        loss = cross_entropy(cls_score, gt_label)
        # compute accuracy
        acc = self.accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        # losses['scores'] = cls_score
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses
    
    def v(self, x, output, gt_label):
        # N, C, H, W = x.size()
        # x = x.reshape(N, C, -1).transpose(2, 1)
        # logits = self.fc(x)
        # x_var = x.var(dim=1).mean(dim=0)
        output['l2norm'] = torch.norm(x, dim=1).mean()
        output['l2normvar'] = torch.norm(x, dim=1).var()
        output['var'] = x.var(dim=1).mean()
        # output['x_varmean'] = x_var.mean()
        # output['x_var'] = x_var.var()
        # y_var = logits.var(dim=1).mean(dim=0)
        # output['y_varmean'] = y_var.mean()
        # output['y_var'] = y_var.var()

        # pred = torch.softmax(logits, dim=2)
        # onehot = torch.zeros(pred.size(0), 10).to("cuda").scatter_(dim=1, index=gt_label.view(-1,1), value=1).view(pred.size(0), 1, 10)
        # pred = (pred * onehot).sum(dim=2)
        # output['pred_var'] = pred.var(dim=1).mean()
        # output['pred_mean'] = pred.mean()
        # output['pred_max'] = pred.max(dim=1)[0].mean()
        # output['pred_min'] = pred.min(dim=1)[0].mean()
        # output['nozero'] = (x != 0).sum(dim=[1, 2]).type(torch.FloatTensor).mean()

    def forward(self, x, gt_label):
        out = self.gap(x).view(x.size(0), -1)

        logits = self.fc(out)
        losses = self.loss(logits, gt_label)

        # self.v(out, losses, gt_label)
        return losses

