import os
import argparse
import torch
import yaml
from models import EffectRegressorMLP
import data


parser = argparse.ArgumentParser("Save categories.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"
device = torch.device(opts["device"])

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.load(opts["save"], "_best", 3)
model.encoder1.eval()
model.encoder2.eval()
model.encoder3.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
X = torch.load("data/img/obs_prev_z.pt")
X = X.reshape(5, 10, 3, 4, 4, 42, 42)
# in test mode, the objects are always in the center
X = X[:, :, 0, 2, 2]
X = X.reshape(-1, 1, 42, 42)
B, _, H, W = X.shape
Y = torch.empty(B, 1, opts["size"], opts["size"])

for i in range(B):
    Y[i] = transform(X[i])

with torch.no_grad():
    category1 = model.encoder1(Y.to(device))
    category2 = model.encoder2(Y.to(device))
category1 = category1.int()
category2 = category2.int()
category_aug = torch.cat([category1, category2], dim=-1)

# I think this takes all combinations of the images and concatenates them
left_img = Y.repeat_interleave(B, 0)
right_img = Y.repeat(B, 1, 1, 1)
concat = torch.cat([left_img, right_img], dim=1)

# this generates the category for the concatenated images
# left ones symbols + right ones symbols + interaction symbols
category3 = model.encoder3(concat.to(device)).int()
left_cat = category_aug.repeat_interleave(B, 0)
right_cat = category_aug.repeat(B, 1)
# TODO Categories should include visible object count at the start
category_all = torch.cat([left_cat, right_cat, category3], dim=-1)
torch.save(category_all.cpu(), os.path.join(opts["save"], "category.pt"))
