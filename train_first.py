import torch
import models
import data
import argparse
import os
import time

parser = argparse.ArgumentParser("train an encoder for effect prediction")
parser.add_argument("-lr", help="learning rate. default 1e-3", default=1e-3, type=float)
parser.add_argument("-bs", help="batch size. default 10", default=10, type=int)
parser.add_argument("-e", help="epoch. default 1000.", default=1000, type=int)
parser.add_argument("-dv", help="device. default cpu.", default="cpu", type=str)
parser.add_argument("-hid", help="hidden size. default 128.", default=128, type=int)
parser.add_argument("-d", help="depth of networks. default 2.", default=2, type=int)
parser.add_argument("-cd", help="code dimension. default 2.", default=2, type=int)
parser.add_argument("-cnn", help="MLP (0) or CNN (1). default 0.", default=0, type=int)
parser.add_argument("-f", help="filters if CNN is used.", nargs="+", type=int)
parser.add_argument("-n", help="batch norm. default 0.", default=0, type=int)
parser.add_argument("-load", help="load model.", type=str)
parser.add_argument("-save", help="save model.", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)

arg_dict = vars(args)
for key in arg_dict.keys():
    print("%s: %s" % (key, arg_dict[key]))
    print("%s: %s" % (key, arg_dict[key]), file=open(os.path.join(args.save, "args.txt"), "a"))
print("date: %s" % time.asctime(time.localtime(time.time())))
print("date: %s" % time.asctime(time.localtime(time.time())), file=(open(os.path.join(args.save, "args.txt"), "a")))

device = torch.device(args.dv)

trainset = data.FirstLevelDataset()
loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True)

if args.cnn == 0:
    encoder = torch.nn.Sequential(
        models.Flatten([1, 2, 3]),
        models.MLP([128*128]+[args.hid]*args.d+[args.cd], normalization="batch_norm" if args.n == 1 else None),
        models.STLayer()
    ).to(device)
else:
    L = len(args.f)-1
    denum = 2**L
    lat = args.f[-1] * ((128 // denum)**2)
    encoder = [models.ConvBlock(
        in_channels=args.f[i],
        out_channels=args.f[i+1],
        kernel_size=3,
        stride=2,
        padding=1,
        batch_norm=True if args.n == 1 else False) for i in range(L)]
    encoder.append(models.Flatten([1, 2, 3]))
    encoder.append(models.MLP([lat, args.cd]))
    encoder.append(models.STLayer())
    encoder = torch.nn.Sequential(*encoder).to(device)

decoder = models.MLP([args.cd + 3] + [args.hid] * args.d + [3]).to(device)
if args.load is not None:
    encoder.load_state_dict(torch.load(os.path.join(args.load, "encoder_first.ckpt")))
    decoder.load_state_dict(torch.load(os.path.join(args.load, "decoder_first.ckpt")))

print("="*10+"ENCODER"+"="*10)
print(encoder)
print("="*27)
print("="*10+"DECODER"+"="*10)
print(decoder)
print("="*27)

optimizer = torch.optim.Adam(
    lr=args.lr,
    params=[
        {"params": encoder.parameters()},
        {"params": decoder.parameters()}
    ],
    amsgrad=True
)
criterion = torch.nn.MSELoss(reduction="sum")
avg_loss = 0.0
it = 0
for e in range(args.e):
    for i, sample in enumerate(loader):
        optimizer.zero_grad()
        st = sample["object"].to(device)
        ac = sample["action"]
        y = sample["effect"].to(device)

        h = encoder(st)
        aug = torch.eye(3, device=h.device)[ac]
        h_bar = torch.cat([h, aug], dim=-1)
        y_bar = decoder(h_bar)

        loss = criterion(y_bar, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        it += 1

    if (e+1) % 100 == 0:
        print("it: %d, loss: %.4f" % (it, avg_loss/it))
        # print("encoder")
        # for p in encoder.parameters():
        #     print(p.data.abs().mean())
        # print("decoder")
        # for p in decoder.parameters():
        #     print(p.data.abs().mean())
        # print("="*30)
        # print(y_bar[0].detach())
        # print(y[0])
        # print("="*30)

with torch.no_grad():
    encoder.eval()
    codes = encoder(trainset.objects.to(device)).cpu()
torch.save(codes, os.path.join(args.save, "codes_first.torch"))
torch.save(encoder.eval().cpu().state_dict(), os.path.join(args.save, "encoder_first.ckpt"))
torch.save(decoder.eval().cpu().state_dict(), os.path.join(args.save, "decoder_first.ckpt"))
