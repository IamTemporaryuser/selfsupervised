num_class = 10
img_norm_cfg = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
transform_train = [[
    dict(type="RandomResizedCrop", size=32, scale=(0.2, 1.0)),
    dict(type="RandomHorizontalFlip"),
    dict(type="ColorJitter", rand_apply=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)],

    [dict(type="RandomResizedCrop", size=32, scale=(0.2, 1.0)),
    dict(type="RandomHorizontalFlip"),
    dict(type="ColorJitter", rand_apply=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)]
]
transform_test = [
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
dataset = dict(
    type='CIFAR10',
    train_root='dataset/cifar10',
    test_root='dataset/cifar10',
    num_workers=64,
    batchsize=512,
    num_class=num_class,
    trainmode="selfsupervisied", 
    transform_train=transform_train, 
    testmode="linear", 
    transform_test=transform_test)

update_interval = 1

total_epochs = 800
optimizer = dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=0.0005, alpha_wd=0)
lr_config = dict(policy='cosine', T_max=total_epochs)

pretrained = None

norm_cfg = dict(type='BN', requires_grad=True)
conv_cfg = dict(type='conv')
neck_norm_cfg = None
zero_init_residual = False

backbone=dict(
        type='ResNet_Cifar',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        zero_init_residual=zero_init_residual)

neck=dict(
    type='ReluNeck',
    in_channels=512,
    frozen_state=False,
    norm_cfg=neck_norm_cfg,
    avgpool=False
)

linear_cfg = dict(type="linear")

head=dict(
   type="Saliency_FC",
   in_channels=512,
   hidden_channels=512,
   out_channels=2048,
   num_features=64,
   num_cls=num_class,
   proj_layers=2,
   avgsize=(1, 1),
   alpha=0.1,
   beta=0.5,
   prob_cls_cfg=dict(just_uniform=False, temperature=0.5, prob_aug=2, batch_size=512, update_interval=update_interval),
   linear_cfg=linear_cfg
)

knn=dict(    
    l2norm=True,
    topk=10
)

logger = dict(interval=50)
saver = dict(interval=400)
evaluate_interval = 10
