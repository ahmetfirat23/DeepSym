import torch
from torchvision import transforms
import numpy as np

# loads the data from data/img folder
# transform optional for image processing
class SingleObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.observation = torch.load("data/img/obs_prev_z.pt").unsqueeze(1)
        self.action = torch.load("data/img/action.pt")

        self.effect = torch.load("data/img/delta_pix_1.pt")
        # normalizes the effects for numerical stability
        self.eff_mu = self.effect.mean(dim=0)
        self.eff_std = self.effect.std(dim=0)
        self.effect = (self.effect - self.eff_mu) / (self.eff_std + 1e-6)

    def __len__(self):
        return len(self.observation)

    # samples are dictionaries with keys "observation", "effect", and "action"
    # if a transform is provided, the observation is transformed
    def __getitem__(self, idx):
        sample = {}
        sample["observation"] = self.observation[idx]
        sample["effect"] = self.effect[idx]
        sample["action"] = self.action[idx]
        if self.transform:
            sample["observation"] = self.transform(self.observation[idx])
        return sample


# loads the data from data/img folder
# transform optional for image processing
class PairedObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # init in train mode
        self.train = True
        self.observation = torch.load("data/img/obs_prev_z.pt")
        self.observation = self.observation.reshape(5, 10, 3, 4, 4, 42, 42)
        self.observation = self.observation[:, :, 0]

        self.effect = torch.load("data/img/delta_pix_3.pt")
        self.effect = self.effect.abs()
        self.eff_mu = self.effect.mean(dim=0)
        self.eff_std = self.effect.std(dim=0)
        self.effect = (self.effect - self.eff_mu) / (self.eff_std + 1e-6)

        self.visible_obj_count_start = torch.load("data/img/visible_obj_count_start.pt")
        self.visible_obj_count_end = torch.load("data/img/visible_obj_count_end.pt")

    def __len__(self):
        return len(self.effect)

    def __getitem__(self, idx):
        sample = {}
        # idx is a number from 0 to 2500
        # obj_i is the first object index
        obj_i = idx // 500
        # size_i is the first object size
        size_i = (idx // 50) % 10
        # obj_j is the second object index
        obj_j = (idx // 10) % 5
        # size_j is the second object size
        size_j = idx % 10
        if self.train:
            # to sample the objects randomly
            ix = np.random.randint(0, 4)
            iy = np.random.randint(0, 4)
            jx = np.random.randint(0, 4)
            jy = np.random.randint(0, 4)
        else:
            # in test mode, the objects are always in the center
            ix, iy, jx, jy = 2, 2, 2, 2

        # get the images of the objects in pairs
        img_i = self.observation[obj_i, size_i, ix, iy]
        img_j = self.observation[obj_j, size_j, jx, jy]
        # apply the transform if provided and stack the images
        if self.transform:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)
            sample["observation"] = torch.cat([img_i, img_j])
        # otherwise, stack the images
        else:
            sample["observation"] = torch.stack([img_i, img_j])
        # get the effect of the pair
        sample["effect"] = self.effect[idx]
        return sample


def default_transform(size, affine, mean=None, std=None):
    transform = [transforms.ToPILImage()]
    if size:
        transform.append(transforms.Resize(size))
    if affine:
        transform.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                fill=int(0.285*255)
            )
        )
    transform.append(transforms.ToTensor())
    if mean is not None:
        transform.append(transforms.Normalize([mean], [std]))
    transform = transforms.Compose(transform)
    return transform
