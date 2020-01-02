from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms
# from scipy.ndimage import rotate, shift
import pickle
from PIL import Image
import torch
import pandas as pd

class Brats19Dataset(Dataset):
    '''
    Dataset designed for Nifti files
    '''
    def __init__(self, root, patient_list=None, labels=True, nifti=False, data_keyword=[], target_keyword=[],
                 data_transform=None, target_transform=None, both_transform=None ):
        '''

        :param root: Data path
        :param labels: Whether labels are there
        :param nifti: Whether return nifti files' name
        :param transform:
        :param target_transform:
        '''
        self.root = root
        self.nifti = nifti
        self.data_list = [[i + '_%s.nii.gz' % keyword for i in patient_list] for keyword in data_keyword]
        self.lesion_list = [[i + '_%s.nii.gz' % keyword for i in patient_list] for keyword in target_keyword]

        self.labels = labels
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.both_transform = both_transform
        self.data = []

    def __getitem__(self, index):

        single_data = [nib.load(os.path.join(self.root, self.data_list[i][index])).get_fdata().astype(np.float32)[np.newaxis, :] for i in range(len(self.data_list))]
        if self.nifti:
            filename = self.data_list[0][index]
        # Load image data from patient. axis is C, Y, X, Z, dtype=float32
        img = np.concatenate(single_data, axis=0)
        if self.labels:
            lesion_data = [nib.load(os.path.join(self.root, self.lesion_list[i][index])).get_fdata().astype(np.int64)[np.newaxis, :] for i in range(len(self.lesion_list))]
            # assert anat_file[:10] == lesion_file[:10]
            target = np.concatenate(lesion_data, axis=0)
            # target[img.sum(axis=0) == 0] = 3
            # target_ = np.zeros_like(target)
            # target = (target != 0).astype(np.int64)  # WT
            # target = (target == 4 ).astype(np.int64)  # ET
            # target = (target == 4).astype(np.int64) + (target == 1).astype(np.int64)  # TC
            # target = (target == 1).astype(np.int64)  # NCR/NEC
            # target = (target == 2).astype(np.int64)  # ED
        else:
            target = None

        if self.data_transform is not None:
            img = self.data_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.both_transform is not None:
            img, target = self.both_transform([img, target])

        if self.labels:
            return img, target
        else:
            if self.nifti:
                return img, filename
            else:
                return img

    def __len__(self):
        return len(self.data_list[0])


class CPMDataset3(Dataset):
    '''
    Dataset designed for Nifti files
    '''
    def __init__(self, root, outdir, patient_list=None, classes=None, labels=True, nifti=False, data_keyword=[],
                 target_keyword=[],
                 data_transform=None, path_transform=None, target_transform=None, both_transform=None ):
        '''

        :param root: Data path
        :param labels: Whether labels are there
        :param nifti: Whether return nifti files' name
        :param transform:
        :param target_transform:
        '''
        self.root = root
        self.nifti = nifti
        self.data_list = [[i + '%s.nii.gz' % keyword for i in patient_list] for keyword in data_keyword]
        self.lesion_list = classes

        self.labels = labels
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.path_transform = path_transform
        self.both_transform = both_transform
        self.patch_names = os.listdir('%s/all_patches/unknow' % outdir)
        self.patch_list = [[os.path.join('%s/all_patches/unknow' % outdir, j) for j in self.patch_names if i in j] for
                           i in
                           patient_list]


    def image_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):

        single_data = [nib.load(os.path.join(self.root, self.data_list[i][index])).get_fdata().astype(np.float32)[np.newaxis, :] for i in range(len(self.data_list))]
        if self.nifti:
            filename = self.data_list[0][index]
        # Load image data from patient. axis is C, Y, X, Z, dtype=float32
        img = np.concatenate(single_data, axis=0)
        # pathology = np.random.choice(self.patch_list[index], 8, True)
        pathology = self.patch_list[index]
        if self.labels:
            if self.lesion_list[index] == 'A':
                target = 0
            elif self.lesion_list[index] == 'G':
                target = 1
            elif self.lesion_list[index] == 'O':
                target = 2
        else:
            target = None

        if self.data_transform is not None:
            img = self.data_transform(img)

        if self.path_transform is not None:
            img2 = [self.path_transform(self.image_loader(path=i)) for i in pathology]
            img2 = torch.stack(img2, dim=0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.both_transform is not None:
            img, target = self.both_transform([img, target])

        if self.labels:
            return img, img2, target
        else:
            if self.nifti:
                return img, img2, filename
            else:
                return img, img2

    def __len__(self):
        return len(self.data_list[0])


class Normalize(object):
    def __init__(self, view='all'):
        if view == 'XY':
            self.axes = (1, 2)
        elif view == 'ZX':
            self.axes = (2, 3)
        elif view == 'ZY':
            self.axes = (1, 3)
        elif view == 'all':
            self.axes = (1, 2, 3)
        else:
            raise TypeError

    def __call__(self, data):
        assert len(data.shape) == 4
        mean = data.mean(axis=self.axes, keepdims=True)
        std = data.std(axis=self.axes, keepdims=True)
        new_data = (data - mean) / (std + 0.000001)

        return new_data


class RandomIntensity(object):
    def __init__(self, prob=0.5, shift=0.1):
        self.prob = prob
        self.shift = shift

    def __call__(self, data):
        if np.random.rand() > self.prob:
            out = np.random.uniform(1-self.shift, 1+self.shift)
            data = data * out

        return data


class PartialRandomIntensity(object):
    def __init__(self, prob=0.5, shift=0.1):
        self.prob = prob
        self.shift = shift

    def __call__(self, data):
        img = data[0]
        et = data[1][3]
        ed = data[1][4]
        ncr = data[1][5]

        if np.random.uniform(0, 1) > 0.5:
            out = np.random.uniform(1-self.shift, 1+self.shift)
            img[:, et == 1] *= out
        if np.random.uniform(0, 1) > 0.5:
            out = np.random.uniform(1 - self.shift, 1 + self.shift)
            img[:, ed == 1] *= out
        if np.random.uniform(0, 1) > 0.5:
            out = np.random.uniform(1-self.shift, 1+self.shift)
            img[:, ncr == 1] *= out

        return img, data[1]


class FlipAug(object):
    def __init__(self, prob=0.5, mode='single'):
        self.prob = prob
        self.mode = mode

    def __call__(self, data):
        if self.mode == 'single':
            new_data = np.concatenate([data, data[:, ::-1, :, :]], axis=0)
            return new_data
        elif self.mode == 'dual':
            img = data[0]
            new_data = np.concatenate([img, img[:, ::-1, :, :]], axis=0)
            return new_data, data[1]


# class RandomRotate(object):
#     def __init__(self, prob=0.5):
#         self.prob = prob
#
#     def __call__(self, data):
#         if np.random.randn() > 0.5:
#             axis = np.random.choice([1, 2, 3], 2, replace=False)
#             degree = np.random.choice(np.arange(15), 1)
#             img = rotate(data[0], angle=degree, axes=axis)
#             label = rotate(data[1], angle=degree, axes=axis)
#
#             return img, label
#         else:
#             return data[0], data[1]


class CropAndPad(object):
    def __init__(self, format_='Brats19', target='data', channel=1):
        if format_ == 'Brats19':
            self.target = target
            if self.target == 'data':
                self.shape = (channel, 160, 192, 160)
            elif self.target == 'label':
                self.shape = (channel, 160, 192, 160)

            self.xstart = 38
            self.xend = 198  # 160 in 0' axis
            self.ystart = 28
            self.yend = 220  # 192 in 1' axis
        else:
            raise AttributeError

    def __call__(self, data):
        if self.target == 'data':
            new_data = np.zeros(shape=self.shape, dtype=np.float32)
            new_data[:, :, :, 5:] = data[:, self.xstart:self.xend, self.ystart:self.yend, :]
        elif self.target == 'label':
            new_data = np.zeros(shape=self.shape, dtype=np.int64)
            new_data[:, :, :, 5:] = data[:, self.xstart:self.xend, self.ystart:self.yend, :]
        return new_data

class RandomCrop(object):
    def __init__(self, format_='Brats19', shape=(160, 192, 128)):
        if format_ == 'Brats19':
            self.shape = shape
            self.size = (240, 240, 155)

    def __call__(self, data):
        x = np.random.choice(np.arange(0, self.size[0] - self.shape[0]))
        y = np.random.choice(np.arange(0, self.size[1] - self.shape[1]))
        z = np.random.choice(np.arange(0, self.size[2] - self.shape[2]))

        img = data[0][:, x:x+self.shape[0], y:y+self.shape[1], z:z+self.shape[2]]
        label = data[1][:, x:x+self.shape[0], y:y+self.shape[1], z:z+self.shape[2]]

        return img, label

class RandomFlip(object):
    def __init__(self, view=None, prob=0.5):
        self.prob = prob
        if view == 'Y':
            self.axes = 1
        elif view == 'X':
            self.axes = 2
        elif view == 'Z':
            self.axes = 3
        else:
            self.axes = None

    def __call__(self, data):
        # assert len(data.shape) == 4
        if np.random.randn() > self.prob:
            if self.axes is None:
                axes = np.random.choice(np.arange(1, 4))
                # data = np.flip(data[0], axis=axes)
                # label = np.flip(data[1], axis=axes - 1)
            else:
                data = np.flip(data[0], axis=self.axes)
                label = np.flip(data[1], axis=self.axes-1)

            return data, label
        else:
            return data


if __name__ == '__main__':
    import pandas as pd

    data_path = '../source_data/training_data_norm'

    data_keyword = ['']
    target_keyword = ['seg', 'wt', 'tc', 'et', 'ed', 'ncr']

    data_transform = transforms.Compose(
        [
            CropAndPad('Brats19', target='data', channel=len(data_keyword)),
            # Normalize(view=config['norm_axis']),
            # RandomIntensity(prob=0.5, shift=0.8),
        ]
    )
    # data_transform=None
    path_transform = transforms.Compose(
        [
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])
    both_transform = None
    inference_transform = transforms.Compose(
        [
            # Normalize(view=config['norm_axis']),
            CropAndPad('Brats19', target='data', channel=len(data_keyword)),

        ]
    )
    label_transform = transforms.Compose(
        [
            CropAndPad('Brats19', target='label', channel=len(target_keyword)),
        ]
    )

    train_list = pd.read_csv('statistic/training_data_classification_labels.csv')['CPM_RadPath_2019_ID']
    test_list = pd.read_csv('statistic/training_data_classification_labels.csv')['CPM_RadPath_2019_ID']

    labels = pd.read_csv('statistic/training_data_classification_labels.csv')['class']
    trainset = CPMDataset3(root=data_path, patient_list=train_list, classes=labels, labels=True, nifti=False,
                           data_keyword=data_keyword,
                           target_keyword=target_keyword,
                           data_transform=data_transform, path_transform=path_transform,
                           both_transform=both_transform)

    train_loader = DataLoader(dataset=trainset, batch_size=2, shuffle=True,
                              num_workers=8, pin_memory=True)

    testset = CPMDataset3(root=data_path, patient_list=test_list, labels=True, nifti=False, data_keyword=data_keyword,
                          target_keyword=target_keyword,
                          data_transform=inference_transform, path_transform=path_transform,
                          target_transform=label_transform)

    test_loader = DataLoader(dataset=testset, batch_size=2, shuffle=False,
                             num_workers=8, pin_memory=True)

    p = iter(train_loader)
    a,b,c = p.next()