import os
import argparse
import time
import yaml
import torch
from models import EffectRegressorMLP, EffectRegressorRNN
import data

parser = argparse.ArgumentParser("Train effect prediction models.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

device = torch.device(opts["device"])

# load the first level data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size1"], shuffle=True)

model = EffectRegressorMLP(opts)
if opts["load"] is not None:
    model.load(opts["load"], ext="", level=1)
    model.load(opts["load"], ext="", level=2)
    model.load(opts["load"], ext="", level=3)
model.print_model(1)
model.train(opts["epoch1"], loader, 1)

# load the best encoder1
model.load(opts["save"], "_best", 1)

# load the second level data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size2"], shuffle=True)
model.print_model(2)
model.train(opts["epoch2"], loader, 2)

# load the best encoder2
model.load(opts["save"], "_best", 2)

# load the third level data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size3"], shuffle=True)
model.print_model(3)
model.train(opts["epoch3"], loader, 3)

# load rnn level data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.SequentialObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size4"], shuffle=True)
model_rnn = EffectRegressorRNN(opts)
model_rnn.load(opts["save"], "_best", 1)
model_rnn.load(opts["save"], "_best", 2)
model_rnn.print_model(3)
model_rnn.train(opts["epoch4"], loader, 3)


