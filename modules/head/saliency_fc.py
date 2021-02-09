import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from ..utils import build_linear_layer
from mmcv.cnn import (normal_init, constant_init, kaiming_init)
from ..apis import train

# def D(p, z, k=1):
#     z = z.detach() # stop gradient
#     sim = torch.bmm(z, p)
#     val, _ = torch.topk(sim, k=k ,dim=1, largest=True)

#     return -val.mean()

def D(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

class BatchNormFC1d(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_features)
        self.init_weights()
    
    def init_weights(self):
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)
    
    def forward(self, x):
        N, L, C = x.size()
        x = x.view(N*L, C)
        out = self.norm(x)
        out = out.view(N, L, C)
        return out

class ParametricLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, norm=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.setproj = False
        if self.setproj:
            self.proj = nn.Linear(in_features=in_features, out_features=in_features)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if norm:
            self.norm = nn.BatchNorm1d(out_features, affine=False)
        else:
            self.norm = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.setproj:
            self.proj.reset_parameters()
            nn.init.constant_(self.proj.bias, 0)

        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        weight = F.normalize(self.weight, dim=1)

        if self.setproj:
            x = self.proj(x)
            
        x = F.normalize(x, dim=-1)

        out = F.linear(x, weight, self.bias)

        if self.norm is not None:
            out = self.norm(out)
        return out

def sinkhorn(Q, eps=0.05, nmb_iters=3):
    with torch.no_grad():
        Q = torch.exp(Q / eps).T
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

class ProbBuffer(nn.Module):
    def __init__(self, classifier, temperature, num_features, batch_size, update_interval):
        super().__init__()
        assert batch_size > 1
        assert update_interval >= 1

        self.C = num_features
        self.num_cls = classifier.out_features
        self.batch_size = batch_size
        self.N = update_interval * batch_size
        self.classifier = classifier
        self.tau = temperature

        self.register_buffer("queue", torch.rand(self.N, self.C))
        self.register_buffer("queue_size", torch.zeros(1, dtype=torch.long))
            
    def forward(self, features, probs, queue_start):
        if torch.distributed.is_initialized():
            rank_index = torch.distributed.get_rank()
            distributed_batch_size = probs.size(0)
            pos_start = rank_index * distributed_batch_size

            features_gather = concat_all_gather(features)
            probs_gather = concat_all_gather(probs)

            others_probs = probs_gather.sum(dim=0) - probs_gather[pos_start:pos_start+distributed_batch_size].sum(dim=0)
            multi_power = torch.distributed.get_world_size()
        else:
            features_gather = features
            probs_gather = probs
            others_probs = 0.0
            multi_power = 1.0

        batch_size = probs_gather.shape[0]
        queue_size = int(self.queue_size)

        if queue_start >= 0:
            uniformity = (probs.sum(dim=0) + others_probs) / batch_size
            return multi_power * (math.log(self.num_cls) + uniformity.log().mean())

        with torch.no_grad():
            self.queue[queue_size:queue_size + batch_size] = features_gather.detach().clone()

            prob_distributions = torch.softmax(self.classifier(self.queue) / self.tau, dim=1)
            channel_wise_prob = torch.sum(prob_distributions[0:queue_size], dim=0)
            # channel_wise_prob = torch.sum(self.queue[0:queue_size], dim=0).cuda()

            queue_size = queue_size + batch_size

            if queue_size == self.N or batch_size != self.batch_size:
                self.queue_size[0] = 0
            else:
                self.queue_size[0] = queue_size
            multi_power = multi_power * queue_size / batch_size
        
        prob_uniform_loss = (channel_wise_prob + others_probs + torch.sum(probs, dim=0)) / queue_size
        prob_uniform_loss = multi_power * (prob_uniform_loss.log().mean() + math.log(self.num_cls))

        return prob_uniform_loss

    def extra_repr(self):
        return 'num_features={}, batch_size={}, num_sample={}'.format(self.C, self.batch_size, self.N)

class ProbLoss(nn.Module):
    def __init__(self, num_features, num_cls, batch_size, update_interval, alpha, beta, just_uniform=False, 
                    temperature=0.5, prob_aug=2, queue_start=0, prob_start=None):
        super().__init__()
        self.C = num_features
        self.tou = temperature
        self.prob_aug = prob_aug
        self.just_uniform = just_uniform
        self.queue_start = queue_start
        self.alpha = alpha
        self.beta = beta
        self.prob_start = prob_start
        self.classifier = nn.Linear(num_features, num_cls)

        normal_init(self.classifier, mean=0, std=0.01, bias=0)

        self.buffer1 = ProbBuffer(self.classifier, temperature, self.C, batch_size, update_interval)
        self.buffer2 = ProbBuffer(self.classifier, temperature, self.C, batch_size, update_interval)

    def forward(self, z1, z2, record_dict=None, prefix=None):
        z1_cls = self.classifier(z1)
        z2_cls = self.classifier(z2)

        p1 = torch.softmax(z1_cls / self.tou, dim=-1)
        p2 = torch.softmax(z2_cls / self.tou, dim=-1)

        q1 = torch.softmax(z1_cls / self.tou * self.prob_aug, dim=-1).detach()
        q2 = torch.softmax(z2_cls / self.tou * self.prob_aug, dim=-1).detach()

        # q1 = sinkhorn(z1, eps=self.tou / 2).detach()
        # q2 = sinkhorn(z2, eps=self.tou / 2).detach()

        prob_cls = - ((q1 * p2.log()).sum(dim=-1).mean() + (q2 * p1.log()).sum(dim=-1).mean()) / 2
        # prob_cls = - ((q1 * p2.log()).mean() + (q2 * p1.log()).mean()) / 2

        prob_uniform = - (self.buffer1(z1, p1, self.queue_start) + self.buffer2(z2, p2, self.queue_start)) / 2

        if record_dict is not None:
            record_dict[f'{prefix}_cls'] = prob_cls
            record_dict[f'{prefix}_uniform'] = prob_uniform
        
        if self.just_uniform:
            return prob_uniform
        else:
            return self.alpha * prob_cls + self.beta * prob_uniform
    
    def set_epoch(self, epoch):
        if epoch >= self.queue_start:
            self.queue_start = -1
        
        if self.prob_start is not None and epoch >= self.prob_start['start']:
            self.prob_aug = self.prob_start['aug']

    def extra_repr(self):
        return 'prob_aug={}, just_uniform={}, queue_start={}, prob_start={}, alpha={}, beta={}'.format(
            self.prob_aug, self.just_uniform, self.queue_start, self.prob_start, self.alpha, self.beta)


class Saliency_FC(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, scale=dict(min=1.0, maxepoch=0),
                    num_features=64, num_cls=10, proj_layers=2, avgsize=(1, 1), linear_cfg=dict(type="linear"), 
                    alpha=0.1, beta=1.0, prob_cls_cfg=dict()):
        super().__init__()
        self.linear_cfg = linear_cfg
        self.num_features = num_features
        self.scale = scale
        self.alpha = alpha
        self.beta = beta

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_bank = ParametricLinear(out_channels, num_features)
        # self.feature_bank_loss = ProbLoss(num_features, **prob_bank_cfg)

        self.feature_cls_loss = ProbLoss(num_features, num_cls, alpha=self.alpha, beta=self.beta, **prob_cls_cfg)

        proj_list = []
        for _ in range(proj_layers-1):
            proj_list.extend(self.__build_layer(in_channels, out_channels, norm=True, relu=True))
            in_channels = out_channels
        proj_list.extend(self.__build_layer(out_channels, out_channels, norm=True, relu=False))
        self.proj = nn.Sequential(*proj_list)

        pred_list = self.__build_layer(out_channels, hidden_channels, norm=True, relu=True)
        pred_list.extend(self.__build_layer(hidden_channels, out_channels, norm=False, relu=False))
        
        self.pred = nn.Sequential(*pred_list)
    
    def __build_layer(self, in_features, out_features, norm=True, relu=True):
        ret = build_linear_layer(self.linear_cfg, in_features, out_features, norm=False)
        if norm:
            ret.append(nn.BatchNorm1d(out_features))
        if relu:
            ret.append(nn.ReLU(inplace=True))
        return ret

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.reset_parameters()

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def v(self, x, output):
        output['l2norm'] = torch.norm(x, dim=[1, 2]).mean()
        output['l2normvar'] = torch.norm(x, dim=[1, 2]).var()
        output['var'] = x.var(dim=[1, 2]).mean()

    def forward(self, x1:torch.Tensor, x2:torch.Tensor):

        def __cal_sim(x1, x2, k):
            z1 = self.proj(x1)
            z2 = self.proj(x2)

            p1 = self.pred(z1)
            p2 = self.pred(z2)
            
            loss_sim = D(p1, z2) / 2 + D(p2, z1) / 2
            return loss_sim, z1, z2
        
        def __cal_scale():
            now_epoch = train.global_epoch
            self.feature_cls_loss.set_epoch(now_epoch)
            if now_epoch >= self.scale['maxepoch']:
                return 1.0
            
            s = self.scale['min'] + (1 - self.scale['min']) * now_epoch / float(self.scale['maxepoch'])
            return s
        
        N, C, H, W = x1.size()
        x1 = self.avgpool(x1).view(N, C)
        x2 = self.avgpool(x2).view(N, C)

        outputs = {}
        # self.v(x1, outputs)

        loss_sim, z1, z2 = __cal_sim(x1, x2, 1)
        outputs['sim'] = loss_sim

        # z1 = z1.squeeze(dim=1)
        # z2 = z2.squeeze(dim=1)

        z1 = self.feature_bank(z1)
        z2 = self.feature_bank(z2)

        now_scale = __cal_scale()

        # loss_prob = self.feature_bank_loss(z1, z2, record_dict=outputs, prefix="bank") + self.feature_cls_loss(z1_cls, z2_cls, record_dict=outputs, prefix="final")
        loss_prob = self.feature_cls_loss(z1, z2, record_dict=outputs, prefix="prob")
        # loss_prob = self.feature_bank_loss(z1, z2, record_dict=outputs, prefix="prob")
        # loss = loss_sim
        loss = loss_sim + now_scale * loss_prob
        outputs['loss'] = loss

        outputs['scale'] = torch.Tensor([now_scale])
        
        return outputs 
    
    def evaluate(self, x):
        N, C, H, W = x.size()
        x = self.avgpool(x).view(N, C)
        x = self.proj(x)
        x = self.feature_bank(x)
        x = self.feature_cls.classifier(x)
        return x

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output