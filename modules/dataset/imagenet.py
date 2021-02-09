import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from .dataset_type import ImageList
from .build import DataPrefetcher
from .build import MultiEpochsDataLoader
from .build import CudaDataLoader
from .pipeline import GaussianBlur
import copy
from .pipeline.transforms import build_transforms

META_FILE = "meta.bin"

class ImageNet_Dataset(data.Dataset):
    def __init__(self, root, transforms, list_file, num_class, preload=False):
        super(ImageNet_Dataset, self).__init__()
        self.root = root
        self.list_file = list_file
        self.num_class = num_class

        self.imagelist = ImageList(root, list_file, num_class, preload)

        self.pipeline = transforms

        if self.imagelist.has_labels:
            self.item_func = self.__get_labeled_item
        else:
            self.item_func = self.__get_unlabeled_item
            if not isinstance(self.pipeline, list):
                self.pipeline = []
                self.pipeline.append(copy.deepcopy(transforms))
                self.pipeline.append(transforms)

    def __len__(self):
        return self.imagelist.get_length()
    
    def __get_labeled_item(self, idx):
        img, target = self.imagelist.get_sample(idx)
        if self.pipeline is not None:
            img = self.pipeline(img)
        
        return img, target

    def __get_unlabeled_item(self, idx):
        img = self.imagelist.get_sample(idx)
        if self.pipeline is not None:
            if isinstance(self.pipeline, list):
                img1 = self.pipeline[0](img)
                img2 = self.pipeline[1](img)
                return img1, img2
    
    def __getitem__(self, idx):
        return self.item_func(idx)

class ImageNet():
    def __init__(self, train_root, test_root, train_list, test_list, batchsize, num_workers, usemultigpu=False, num_class=1000, 
                        trainmode="linear", transform_train=None, testmode="linear", transform_test=None):
        
        assert transform_train is not None and transform_test is not None

        self.transform_train = build_transforms(transform_train)
        self.transform_test = build_transforms(transform_test)
        
        self.usemultigpu = usemultigpu
        
        self.trainset = ImageNet_Dataset(train_root, self.transform_train, train_list, num_class, preload=False)

        self.testset = ImageNet_Dataset(test_root, self.transform_test, test_list, num_class, preload=False)

        if self.usemultigpu:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
        else:
            self.train_sampler = None
        self.test_sampler = None
        
        self.train_loader = DataPrefetcher(torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, 
                                shuffle=not self.usemultigpu, num_workers=num_workers, pin_memory=True, sampler=self.train_sampler))

        self.test_loader = DataPrefetcher(torch.utils.data.DataLoader(self.testset, batch_size=batchsize, 
                                shuffle=False, num_workers=num_workers, pin_memory=True, sampler=self.test_sampler))

        # self.train_loader = CudaDataLoader(
        #     MultiEpochsDataLoader(self.trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True),
        #     "cuda", queue_size=8)
        
        # self.test_loader = CudaDataLoader(
        #     MultiEpochsDataLoader(self.testset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True),
        #     "cuda", queue_size=8)

    def get_loader(self):
        return self.train_loader, self.test_loader

    def set_epoch(self, epoch):
        if self.usemultigpu:
            self.train_sampler.set_epoch(epoch)

    def testsize(self):
        return self.testset.__len__()
    
    def trainsize(self):
        return self.trainset.__len__()
