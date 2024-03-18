import numpy as np
import torch

from torchvision.datasets import ImageFolder


# https://discuss.pytorch.org/t/data-loader-for-triplet-loss-cross-entropy-loss/102896/3
class TripletImageFolder(ImageFolder):
    """From the torchvision.datasets.ImageFolder it generates triplet samples, used in training. For testing we use normal image folder.
    Note: a triplet is composed by a pair of matching images and one of different class.
    """

    def __init__(self, *arg, **kw):
        super(TripletImageFolder, self).__init__(*arg, **kw)

        self.n_triplets = len(self.samples)
        self.train_triplets = self.generate_triplets()

    def generate_triplets(self):
        labels = torch.Tensor(self.targets)
        triplets = []
        for x in np.arange(self.n_triplets):
            idx = np.random.randint(0, labels.size(0))
            idx_matches = np.where(labels.numpy() == labels[idx].numpy())[0]
            idx_no_matches = np.where(labels.numpy() != labels[idx].numpy())[0]
            idx_a, idx_p = np.random.choice(idx_matches, 2, replace=False)
            idx_n = np.random.choice(idx_no_matches, 1)[0]
            triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)

    def set_triplets(self, triplets):
        self.train_triplets = triplets

    def __getitem__(self, index):
        t = self.train_triplets[index]

        path_a, _ = self.samples[t[0]]
        path_p, _ = self.samples[t[1]]
        path_n, _ = self.samples[t[2]]

        img_a = self.loader(path_a)
        img_p = self.loader(path_p)
        img_n = self.loader(path_n)

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n
