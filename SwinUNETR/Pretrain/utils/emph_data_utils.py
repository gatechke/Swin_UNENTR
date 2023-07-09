# v3: used to calculate recall/precision, patch size 64, *with* normal samples, no normalization for loc

from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
import os, glob

PATCH_SIZE = 32


def default_transform(x):
    return x


def random_crop_transform(x):
    loc = np.random.randint(low=0, high=33, size=3)
    return x[:, loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE, loc[2]:loc[2] + PATCH_SIZE], loc


def center_crop_transform(x):
    return x[:, PATCH_SIZE // 2:PATCH_SIZE * 3 // 2, PATCH_SIZE // 2:PATCH_SIZE * 3 // 2,
           PATCH_SIZE // 2:PATCH_SIZE * 3 // 2]


class COPD_dataset(Dataset):

    def __init__(self, target, transforms=default_transform, stage='train', fold=0, fraction=1.):
        self.metric_dict = dict()  # initialize metric dictionary
        self.transforms = transforms
        patch_loc = np.load("/ocean/projects/asc170022p/lisun/copd/gnn_shared/3D-SSL/emph/atlas_coord_v2.npy")
        self.patch_list = []
        self.loc_list = []
        self.label_list = []
        print("Target:", target)

        FILE = open(
            "/ocean/projects/asc170022p/shared/Data/COPDGene/ClinicalPatches/LearningEmphysemaFromVisualPatches/listEmphysemaSamples.csv",
            "r")
        FILE.readline()
        for line in FILE.readlines():
            mylist = line.strip("\n").split(",")
            mylist[2] = mylist[2].lower()
            if target not in mylist[2]:
                continue

            # Normal patches are from matched samples
            if os.path.exists("/ocean/projects/asc170022p/lisun/copd/gnn_shared/3D-SSL/emph/patch_match_v3/" + mylist[3] + ".npy"):
                self.patch_list.append(np.load("/ocean/projects/asc170022p/lisun/copd/gnn_shared/3D-SSL/emph/patch_match_v3/" + mylist[3] + ".npy"))
                self.loc_list.append(patch_loc[int(mylist[3]) - 1])
                self.label_list.append(0)

            if os.path.exists("/ocean/projects/asc170022p/lisun/copd/gnn_shared/3D-SSL/emph/patch_v2/" + mylist[3] + ".npy"):
                self.patch_list.append(np.load("/ocean/projects/asc170022p/lisun/copd/gnn_shared/3D-SSL/emph/patch_v2/" + mylist[3] + ".npy"))
                self.loc_list.append(patch_loc[int(mylist[3]) - 1])
                # binary class
                if "mild" in mylist[2]:
                    self.label_list.append(1)
                elif "moderate" in mylist[2]:
                    self.label_list.append(1)
                elif "severe" in mylist[2]:
                    self.label_list.append(1)

        self.loc_list = np.stack(self.loc_list).astype(np.int)
        fixed_patch_loc = np.load(
            "/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/patch_data_32_6_reg_mask/19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.fixed_patch_loc_max = fixed_patch_loc.max(0)
        self.loc_list = self.loc_list  # / fixed_patch_loc.max(0) * 2. - 1. # [-1,1]
        print("loc_list:", self.loc_list.shape)
        self.label_list = np.array(self.label_list).astype(np.int)
        self.patch_list = np.stack(self.patch_list).astype(np.int)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        kf_fold = list(kf.split(self.patch_list))
        train_index, valid_index = kf_fold[fold]

        if stage == 'train':
            self.patch_list = self.patch_list[train_index]
            self.label_list = self.label_list[train_index]
            self.loc_list = self.loc_list[train_index]
            if fraction < 1:
                num_sample = int(fraction * self.patch_list.shape[0])
                np.random.seed(0)
                sample = np.random.choice(self.patch_list.shape[0], size=num_sample, replace=False)
                self.patch_list = self.patch_list[sample]
                self.label_list = self.label_list[sample]
                self.loc_list = self.loc_list[sample]
        else:
            self.patch_list = self.patch_list[valid_index]
            self.label_list = self.label_list[valid_index]
            self.loc_list = self.loc_list[valid_index]

        print("Dataset size:", len(self))

    def __len__(self):
        return self.loc_list.shape[0]

    def get_labels(self):
        return self.label_list

    def __getitem__(self, idx):
        img = self.patch_list[idx:idx + 1]
        img = img + 1024.
        pack = self.transforms(img)
        if len(pack) == 2:
            img, loc = pack
            img = img / 632. - 1  # Normalize to [-1,1], 632=(1024+240)/2
            loc = loc - 16
            patch_loc = self.loc_list[idx] + np.array([loc[2], loc[1], loc[0]])
            patch_loc = patch_loc / self.fixed_patch_loc_max * 2. - 1.  # [-1,1]
        else:
            img = pack
            img = img / 632. - 1  # Normalize to [-1,1], 632=(1024+240)/2
            patch_loc = self.loc_list[idx]
        return img, patch_loc, self.label_list[idx]


# test code for the dataset
emph_dataset = COPD_dataset(target="emphysema", transforms=random_crop_transform, stage='train', fold=0, fraction=1.)
print(emph_dataset[0][0].shape)

# test code for dataloader enumerate
from torch.utils.data import DataLoader
emph_dataloader = DataLoader(emph_dataset, batch_size=1, shuffle=True, num_workers=0)
for i, (img, loc, label) in enumerate(emph_dataloader):
    print(i, img.shape, loc.shape, label.shape)
    break