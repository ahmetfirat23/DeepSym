import torch
import numpy as np
from sklearn.tree import _tree
from torch.distributions import Gumbel


def kmeans(x, k, centroids=None, max_iter=None, epsilon=0.01):
    """
    x: data set of size (n, d) where n is the sample size.
    k: number of clusters
    centroids (optional): initial centroids
    max_iter (optional): maximum number of iterations
    epsilon (optional): error tolerance
    returns
    centroids: centroids found by k-means algorithm
    next_assigns: assignment vector
    mse: mean squared error
    """
    if centroids is None:
        centroids = torch.zeros(k, x.shape[1], device=x.device)
        prev_assigns = torch.randint(0, k, (x.shape[0],), device=x.device)
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)

    distances = torch.cdist(centroids, x) ** 2
    prev_assigns = torch.argmin(distances, dim=0)

    it = 0
    prev_mse = distances[prev_assigns, torch.arange(x.shape[0])].mean()
    while True:
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)
        distances = torch.cdist(centroids, x) ** 2
        next_assigns = torch.argmin(distances, dim=0)
        if (next_assigns == prev_assigns).all():
            break
        else:
            prev_assigns = next_assigns
        it += 1
        mse = distances[next_assigns, torch.arange(x.shape[0])].mean()
        error = abs(prev_mse-mse)/prev_mse
        prev_mse = mse
        print("iteration: %d, mse: %.3f" % (it, prev_mse.item()))

        if it == max_iter:
            break
        if error < epsilon:
            break

    return centroids, next_assigns, prev_mse, it


def tree_to_code(tree, feature_names, effect_names, obj_names, below_larger):
    tree_ = tree.tree_

    def recurse(node, rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            left = rules.copy()
            right = rules.copy()
            left.append(-(tree_.feature[node]+1))
            right.append(tree_.feature[node]+1)
            rules_from_left = recurse(tree_.children_left[node], left)
            rules_from_right = recurse(tree_.children_right[node], right)
            rules = np.concatenate([rules_from_left, rules_from_right])
            return rules
        else:
            absrules = np.abs(rules).tolist()
            indices = [absrules.index(x) for x in range(1, 6)]

            if rules[indices[0]] != 0 and rules[indices[1]] != 0:
                obj1 = obj_names[(np.sign(rules[indices[0]]), np.sign(rules[indices[1]]))]
            else:
                obj1 = ""
                print("Error 1")

            if rules[indices[2]] != 0 and rules[indices[3]] != 0:
                obj2 = obj_names[(np.sign(rules[indices[2]]), np.sign(rules[indices[3]]))]
            else:
                obj2 = ""
                print("Error 2")

            if rules[indices[4]] != 0:
                if below_larger == np.sign(rules[indices[4]]):
                    comparison = "(larger ?below ?above)"
                else:
                    comparison = "(larger ?above ?below)"
            else:
                comparison = ""
                print("Error 3")

            precond = ":precondition (and (pickloc ?above) (stackloc ?below) "
            precond += "(%s ?above) (%s ?below) %s)" % (obj1, obj2, comparison)
            print(rules)
            print(precond)
            eff = tree_.value[node][0]
            effect = ":effect (and (probabilistic"
            # this shenanigan is needed because probabilities add up to more than one.
            probs = (eff / eff.sum())
            probs = (probs * 1000).round().astype(np.int)
            ptotal = probs.sum()
            if ptotal > 1000:
                residual = ptotal - 1000
                probs[np.argmax(probs)] -= residual

            for i in range(len(eff)):
                if probs[i] != 1000:
                    effect += "\n\t\t\t\t 0.%03d " % (probs[i])
                else:
                    effect += "\n\t\t\t\t 1.000 "

                if effect_names[i] == "stacked" or effect_names[i] == "inserted":
                    effect += "(and (%s) (instack ?above) (stackloc ?above) (not (stackloc ?below)))" % effect_names[i]
                else:
                    effect += "(%s)" % (effect_names[i])
            effect += ")"
            effect += "\n\t\t\t\t(not (pickloc ?above)))"
            return np.array([[precond, effect]])
    return recurse(0, [])


def gumbel_softmax_sample(logits, temp=1.):
    g = Gumbel(0, 1).sample(logits.shape)
    y = (g + logits) / temp
    return torch.softmax(y, dim=-1)


def gumbel_softmax(logits, temp=1.):
    y = gumbel_softmax_sample(logits, temp)
    ind = torch.argmax(y, dim=-1)
    y_hard = torch.eye(logits.shape[-1], device=logits.device)[ind]
    y = (y_hard - y).detach() + y
    return y


def get_parameter_count(model):
    total_num = 0
    for p in model.parameters():
        shape = p.shape
        num = 1
        for d in shape:
            num *= d
        total_num += num
    return total_num


def return_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def cc_pix_avg(img, x, y):
    height, width = img.shape
    img[x, y] = False
    painted = [[x, y]]
    if x+1 < height and img[x+1, y]:
        painted += cc_pix_avg(img, x+1, y)
    if x-1 > 0 and img[x-1, y]:
        painted += cc_pix_avg(img, x-1, y)
    if y+1 < width and img[x, y+1]:
        painted += cc_pix_avg(img, x, y+1)
    if y-1 < 0 and img[x, y-1]:
        painted += cc_pix_avg(img, x, y-1)
    return painted


def find_objects(img, window_size):
    height, width = img.shape
    half_window = window_size // 2
    objects = []
    locations = []
    ground = img.max()
    mask = img < (img.min() + 0.01)
    is_empty = mask.all()
    while not is_empty:
        h_i, w_i = mask.nonzero()[0]
        pp = cc_pix_avg(mask, h_i.item(), w_i.item())
        h_c, w_c = np.mean(pp, axis=0).round().astype(np.int)
        locations.append([h_c, w_c])
        h_c = np.clip(h_c, half_window, width-half_window)
        w_c = np.clip(w_c, half_window, width-half_window)
        objects.append(img[(h_c-half_window):(h_c+half_window), (w_c-half_window):(w_c+half_window)].clone())
        img[(h_c-half_window):(h_c+half_window), (w_c-half_window):(w_c+half_window)] = ground
        mask = img < (img.min()+0.01)
        is_empty = mask.all()
    return torch.stack(objects), torch.tensor(locations)
