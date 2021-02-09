num_class = 10
img_norm_cfg = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
transform_train = [
    dict(type="RandomResizedCrop", size=32, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=3),
    dict(type="RandomHorizontalFlip"),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
transform_test = [
    dict(type="Resize", size=int(32 * (8 / 7)), interpolation=3), #Image.BICUBIC
    dict(type="CenterCrop", size=32),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]

dataset = dict(
    type='CIFAR10',
    train_root='dataset/cifar10',
    test_root='dataset/cifar10',
    num_workers=32,
    batchsize=256,
    num_class=num_class,
    trainmode="linear", 
    transform_train=transform_train, 
    testmode="linear", 
    transform_test=transform_test)

total_epochs = 100
optimizer = dict(type='SGD', lr=30, momentum=0.9, weight_decay=0, alpha_wd=0)
# lr_config = dict(policy='cosine', T_max=total_epochs)
lr_config = dict(policy='step', milestones=[60, 80])

pretrained = None

use_pws = False
norm_cfg = dict(type='BN', requires_grad=True)
conv_cfg = dict(type='conv')
neck_norm_cfg = None
zero_init_residual = False
if use_pws:
    norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
    conv_cfg = dict(type='pws', gamma=1e-3, equiv=False, initalpha=True, mode="fan_in")
    neck_norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
    zero_init_residual = False

backbone=dict(
        type='ResNet_Cifar',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=4,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        zero_init_residual=zero_init_residual)

neck=dict(
    type='ReluNeck',
    in_channels=512,
    frozen_state=True,
    norm_cfg=neck_norm_cfg,
    avgpool=False
)

head=dict(
    type='SaliencyCLSHead',
    num_classes=num_class,
    in_channels=512,
    topk=(1, 5),
)
# head=dict(
#    type="SaliencyCLSHead",
#    in_channels=512,
#    hidden_channels=512,
#    out_channels=2048,
#    num_classes=num_class
# )


logger = dict(interval=50)
saver = dict(interval=100)
