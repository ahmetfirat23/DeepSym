import torch
from torchvision import transforms
import numpy as np

# loads the data from data/img folder
# transform optional for image processing
class SingleObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.observation = torch.load("data/img/obs_1.pt").unsqueeze(1)
        self.action = torch.load("data/img/action_1.pt")

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

# Load data for second level. This is the same as SingleObjectData but with different data files.
# This data is used to learn the effect of flip action.
class SingleFlipObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.observation = torch.load("data/img/obs_2.pt").unsqueeze(1)

        self.effect = torch.load("data/img/delta_pix_2.pt")
        self.eff_mu = self.effect.mean(dim=0)
        self.eff_std = self.effect.std(dim=0)
        self.effect = (self.effect - self.eff_mu) / (self.eff_std + 1e-6)

    def __len__(self):
        return len(self.observation)


# loads the data from data/img folder
# transform optional for image processing
class PairedObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.observation = torch.load("data/img/obs_3.pt")
        # TODO update the datasizes to match the new data
        self.observation = self.observation.reshape(-1, 2, 42, 42)

        self.action = torch.load("data/img/action_3.pt")
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
        # randomize the order of the images
        i, j = np.random.choice([0,1], 2, replace=False)
        img_i = self.observation[idx, i]
        img_j = self.observation[idx, j]
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
        sample["action"] = self.action[idx]
        sample["visible_obj_count_start"] = self.visible_obj_count_start[idx]
        sample["visible_obj_count_end"] = self.visible_obj_count_end[idx]
        return sample


class SequentialObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.observation = torch.load("data/img/obs_4.pt")
        self.observation = self.observation.reshape(2500, -1, 42, 42) # Assume 2500 sequences of 42x42 images

        self.action = torch.load("data/img/action_4.pt")
        self.effect = torch.load("data/img/delta_pix_4.pt")
        self.eff_mu = self.effect.mean(dim=0)
        self.eff_std = self.effect.std(dim=0)
        self.effect = (self.effect - self.eff_mu) / (self.eff_std + 1e-6)

    def __len__(self):
        return len(self.effect)
    
    def __getitem__(self, idx):
        sample = {}
        observations = self.observation[idx]
        actions = self.action[idx]
        effects = self.effect[idx]

        if self.transform:
            for i in range(len(observations)):
                observations[i] = self.transform(observations[i])
        
        # If second object is going to be showed to the model, change the order of the objects to prevent the model from learning the order
        for i in range(len(observations)):
            no_action = actions[-1]
            if no_action == 1:
                p = np.random.randint(0, 2)
                if p == 1:
                    actions[i-1], actions[i] = actions[i], actions[i-1]
                    observations[i-1], observations[i] = observations[i], observations[i-1]
                    effects[i-1], effects[i] = effects[i], effects[i-1]

        sample["observation"] = observations
        sample["effect"] = effects
        sample["action"] = actions
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