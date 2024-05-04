import math
import os
import torch

# gumbel sigmoid function for binary representation
# hard version
class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp(-1., 1.)

# gumbel sigmoid layer
class STLayer(torch.nn.Module):

    def __init__(self):
        super(STLayer, self).__init__()
        self.func = StraightThrough.apply

    def forward(self, x):
        return self.func(x)

# linear layer
# batch normalization optional
class Linear(torch.nn.Module):
    """ linear layer with optional batch normalization. """
    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        # normal weights and biases with 0 mean and std is selected
        # if std is given
        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        # if gain is given, we use the gain / weight size to initialize the weight
        # bias is zero
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            x = self.batch_norm(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)

# multi-layer perceptron
# batch normalization optional
# input and hidden layer dropout optional
# activation function is ReLU
# layer_info is a list of integers, where the first element is the input dimension and
# preceeding elements are hidden layer dimensions and the last element is the output dimension
# std is the standard deviation for the weight initialization
    
# in drop is the dropout rate for the input layer
# hid drop is the dropout rate for the hidden layers
class MLP(torch.nn.Module):
    """ multi-layer perceptron with batch norm option """
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)

"""convolutional block
batch normalization optional
std is the standard deviation for the weight initialization
bias is optional
in channels are the input channels to the convolutional layer
out channels are the output channels of the convolutional layer
kernel size is the size of the kernel
stride is the stride of the convolution
padding is the padding of the convolution"""
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.block = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        self.block.append(torch.nn.ReLU())

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)

"""Single RNNCell
input size is the size of the symbol and action
hidden size is the size of the hidden state
output size is the size of the 4 (dx, dy, dd, and dF)"""   
class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity="relu"):
        super(RNNCell, self).__init__()
        self.rnn = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=output_size)


    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.shape[0], self.rnn.hidden_size, device=x.device)
        h = self.rnn(x, h)
        return self.linear(h), h

"""RNN
input size is the size of the symbol and action
hidden size is the size of the hidden state
output size is the size of the 4 (dx, dy, dd, and dF)"""  
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity="relu"):
        super(RNN, self).__init__()
        self.cell = RNNCell(input_size, hidden_size, output_size, nonlinearity)

    def forward(self, x):
        h = None
        outputs = []
        for i in range(x.shape[1]):
            y, h = self.cell(x[:, i], h)
            outputs.append(y)
        return torch.stack(outputs, dim=1)


""" flatten layer
 dims is a list of dimensions to flatten
 for example, if dims=[1, 2, 3], then the tensor is flattened along the 1, 2, and 3 dimensions
 if dims=[1, 2], then the tensor is flattened along the 1 and 2 dimensions
 if dims=[1], then the tensor is flattened along the 1 dimension """
class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"

""" average layer
in deepsym used as nn.AdaptiveAvgPool2d((1, 1))
wrapper for torch.mean
dims is a list of dimensions to average
for example, if dims=[1, 2, 3], then the tensor is averaged along the 1, 2, and 3 dimensions
if dims=[1, 2], then the tensor is averaged along the 1 and 2 dimensions
if dims=[1], then the tensor is averaged along the 1 dimension """
class Avg(torch.nn.Module):
    def __init__(self, dims):
        super(Avg, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


def build_encoder(opts, level):
    # code dim is the output sizes of mlp
    # single object encoder
    if level == 1:
        code_dim = opts["code1_dim"]
    # pair object encoder
    elif level == 2:
        code_dim = opts["code2_dim"]
    else:
        code_dim = opts["code3_dim"]
    # if cnn is true, then we use a convolutional encoder for image data
    if opts["cnn"]:
        # get the number of layers according to single or pair object
        L = len(opts["filters"+str(level)])-1
        encoder = []
        for i in range(L):
            # batch normalization comes from opts.yaml
            # first conv layer with stride 1
            encoder.append(ConvBlock(in_channels=opts["filters"+str(level)][i],
                                     out_channels=opts["filters"+str(level)][i+1],
                                     kernel_size=3, stride=1, padding=1, batch_norm=opts["batch_norm"]))
            # second conv layer with stride 2
            encoder.append(ConvBlock(in_channels=opts["filters"+str(level)][i+1],
                                     out_channels=opts["filters"+str(level)][i+1],
                                     kernel_size=3, stride=2, padding=1, batch_norm=opts["batch_norm"]))
        # average every channel so every channel provides a single value
        # this gives one dimensional vector
        encoder.append(Avg([2, 3]))
        # mlp takes array here, first one gives size of the input
        # code_dim gives the sizes of hidden layers
        encoder.append(MLP([opts["filters"+str(level)][-1], code_dim]))
        # discrete representation
        encoder.append(STLayer())

    # otherwise, we use linear layers for the encoder for image data 
        # LOL don't even read this
    else:
        encoder = [
            Flatten([1, 2, 3]),
            MLP([[opts["size"]**2] + [opts["hidden_dim"]]*opts["depth"] + [code_dim]],
                batch_norm=opts["batch_norm"]),
            STLayer()]

    # convert list of layers to sequential model
    encoder = torch.nn.Sequential(*encoder)
    return encoder
