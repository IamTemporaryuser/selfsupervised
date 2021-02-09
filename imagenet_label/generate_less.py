import os
import numpy as np
import random

cls_num = 500
data_percent = 0.1
train_path = "train_labeled.txt"
test_path = "val_labeled.txt"

dst_root = f"less_500"

def generate_dataset(data_lines, test_lines, cls_idx, cls_num, data_percent, dst_root):
    select_idx = random.sample(cls_idx, cls_num)
    random.shuffle(select_idx)

    result = {}
    for l in data_lines:
        img_name = l.split()[0]
        img_cls = int(l.split()[1])
        if img_cls in select_idx:
            if img_cls not in result:
                result[img_cls] = []
            result[img_cls].append([img_name, img_cls])

    now_id = 0

    w_labeled = open(f"{dst_root}/train_labeled.txt", "w")
    w_raw = open(f"{dst_root}/train.txt", "w")

    redirect_id = dict()

    for cls_id in select_idx:
        redirect_id[cls_id] = now_id

        select_cls_idx = random.sample(result[cls_id], int(len(result[cls_id]) * data_percent))
        for r in select_cls_idx:
            w_labeled.write("{} {}\n".format(r[0], now_id))
            w_raw.write("{}\n".format(r[0]))
        
        now_id += 1

    w_labeled.close()
    w_raw.close()

    t_labeled = open(f"{dst_root}/test_labeled.txt", "w")
    t_raw = open(f"{dst_root}/test.txt", "w")

    for l in test_lines:
        img_name, img_cls = l.split()
        img_cls = int(img_cls)

        if img_cls in select_idx:
            t_labeled.write("{} {}\n".format(img_name, redirect_id[img_cls]))
            t_raw.write("{}\n".format(img_name))

if os.path.exists(dst_root):
    raise ValueError(f"dst_root={dst_root} is exist.")

os.mkdir(dst_root)

with open(train_path, "r") as r:
    lines = r.readlines()

with open(test_path, "r") as r:
    test_lines = r.readlines()

print(len(lines))
cls_bin = np.zeros((1000), dtype=np.uint32)
cls_idx = np.arange(1000).tolist()
for l in lines:
    img_cls = int(l.split()[1])
    cls_bin[img_cls] += 1

generate_dataset(lines, test_lines, cls_idx, cls_num, data_percent, dst_root)