import torch
import torch.nn as nn
import numpy as np
import time
import mmcv
from ..utils import KNN
try:
    from sklearn.neighbors import KNeighborsClassifier
except:
    print('doesnt has sklearn')


class evaluate_cls(object):
    def __init__(self, model, device, data_loader, datasize, logger=None):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.logger = logger

    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        correct = None
        total = 0
        prog_bar = mmcv.ProgressBar(self.datasize)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs, targets)

                total += targets.size(0)
                if correct is None:
                    correct = {}
                    for k in outputs['accuracy']:
                        correct[k] = outputs['accuracy'][k].item()
                else:
                    for k in outputs['accuracy']:
                        correct[k] += outputs['accuracy'][k].item()

                for _ in range(targets.size(0)):
                    prog_bar.update()

        for k in correct:
            correct[k] = round(correct[k] / float(total) * 100.0, 4)
        
        self.logger.record_eval(epoch, correct)

class evaluate_knn(object):
    def __init__(self, model, device, data_loader, datasize, config, logger):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger
        self.outputs_container = None
        self.targets_container = np.zeros([1, datasize], dtype=np.int64)
        
        self.knn_config = config.get('knn', {})
        if "topk" not in self.knn_config:
            if "topk_percent" not in self.knn_config:
                self.topk = int(datasize / config['num_class'] * 0.2)
            else:
                self.topk = int(datasize / config['num_class'] * self.knn_config.pop('topk_percent'))
            self.knn_config['topk'] = self.topk
        else:
            self.topk = self.knn_config['topk']
    
    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(self.datasize)

        sample_idx = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                batchsize = targets.size(0)

                outputs = self.model(inputs, forward_knn=True)
                if len(outputs.size()) > 2:
                    outputs = outputs.mean(axis=[2,3])
                    
                if self.outputs_container is None:
                    self.outputs_container = np.zeros([self.datasize, outputs.size(1)], dtype=np.float32)
                
                self.outputs_container[sample_idx:sample_idx+batchsize] = outputs.cpu().numpy()
                self.targets_container[0, sample_idx:sample_idx+batchsize] = targets.cpu().numpy()

                sample_idx += batchsize
                for _ in range(batchsize):
                    prog_bar.update()

        print('==> Calculating KNN..')
        total_acc = KNN(self.outputs_container, self.targets_container, **self.knn_config)
        correct = {
            f"KNN-{self.topk}":total_acc * 100.0
        }
        self.logger.record_eval(epoch, correct)

class evaluate_nocls(object):
    def __init__(self, model, device, data_loader, datasize, config, logger):
        self.model = model
        self.device = device
        self.train_loader, self.test_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger
        self.outputs_container = np.zeros([datasize], dtype=np.int64)
        self.targets_container = np.zeros([datasize], dtype=np.int64)
        
        # self.nn_config = 20
    
    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(self.datasize)

        sample_idx = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                batchsize = targets.size(0)

                outputs = self.model.forward_knn(inputs)
                pred = torch.softmax(outputs, dim=-1)
                _, cls_label = torch.topk(pred, k=1, dim=1, largest=True)
                
                self.outputs_container[sample_idx:sample_idx+batchsize] = cls_label.cpu().squeeze(dim=1).numpy()
                self.targets_container[sample_idx:sample_idx+batchsize] = targets.cpu().numpy()

                sample_idx += batchsize
                for _ in range(batchsize):
                    prog_bar.update()

        print('==> Calculating..')
        num_cls = self.targets_container.max() + 1
        clsset = []
        redirect = dict()
        for c in range(num_cls):
            clsnum = self.targets_container[self.outputs_container == c]
            r = dict()
            for n in clsnum:
                if n not in r:
                    r[n] = 0
                r[n] += 1
            clsset.append(r)

            maxitem = sorted(r.items(), key=lambda x:x[1])[-1]
            redirect[c] = maxitem[0]

            # assert maxitem[1] >= self.nn_config / 2.
            # assert maxitem[0] not in redirect
            # for m in maxitem:
            #     if m not in redirect:
            #         redirect[m] = c
            #         break

        print(clsset)
        print(redirect)

        def mp(entry):
            return redirect[entry]
        mp = np.vectorize(mp)

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.cpu().numpy()
                
                batchsize = targets.shape[0]

                outputs = self.model.forward_knn(inputs)
                pred = torch.softmax(outputs, dim=-1)
                _, cls_label = torch.topk(pred, k=1, dim=1, largest=True)

                total += batchsize
                correct += (mp(cls_label.cpu().squeeze(dim=1).numpy()) == targets).sum()

        total_acc = correct / total
        correct = {
            f"top-1":total_acc * 100.0
        }
        self.logger.record_eval(epoch, correct)

class traditional_knn(object):
    def __init__(self, model, device, data_loader, datasize, config, logger):
        self.model = model
        self.device = device
        self.train_loader, self.test_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger

        self.nums = 20
        self.num_cls = 10
    
    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(self.datasize)

        classifier = KNeighborsClassifier(5, weights="distance")
        X = None
        Y = None
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                batchsize = targets.size(0)

                outputs = self.model.backbone(inputs)
                outputs = self.model.neck(outputs)
                if len(outputs.size()) > 2:
                    outputs = outputs.mean(axis=[2,3])

                if X is None:
                    X = outputs.cpu().detach().clone().numpy()
                    Y = targets.cpu().detach().clone().numpy()
                else:
                    X = np.concatenate([X, outputs.cpu().detach().clone().numpy()], axis=0)
                    Y = np.concatenate([Y, targets.cpu().detach().clone().numpy()], axis=0)
                for _ in range(batchsize):
                    prog_bar.update()

        print('==> Calculating..{}'.format(X.shape))
        percent_x = None
        percent_y = None
        for c in range(self.num_cls):
            if percent_x is None:
                percent_x = X[Y == c][0:self.nums]
                percent_y = Y[Y == c][0:self.nums]
            else:
                percent_x = np.concatenate([percent_x, X[Y == c][0:self.nums]], axis=0)
                percent_y = np.concatenate([percent_y, Y[Y == c][0:self.nums]], axis=0)

        classifier.fit(percent_x, percent_y)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.cpu().numpy()
                
                batchsize = targets.shape[0]
                outputs = self.model.backbone(inputs)
                outputs = self.model.neck(outputs)
                if len(outputs.size()) > 2:
                    outputs = outputs.mean(axis=[2,3])
                predits = classifier.predict(outputs.cpu().numpy())

                total += batchsize
                correct += (predits == targets).sum()

        total_acc = correct / total
        correct = {
            f"top-1":total_acc * 100.0
        }
        self.logger.record_eval(epoch, correct)