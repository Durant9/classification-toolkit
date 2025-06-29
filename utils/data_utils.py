from ast import Raise
import pickle
from PIL import Image
import math as m
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from skimage.util import random_noise
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import os
from collections import defaultdict
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def load_assert_data_file(filepath):
    """
    Check if data file is in this format:
        {
            0: [Image11, Image12, ...],
            1: [Image21, Image22, ...],
            ...
        }
    Where:
        - keys are integers (class labels)
        - values are lists of PIL.Image.Image objects
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    assert isinstance(data, dict), "Data file must be a dict."

    for k, v in data.items():
        assert isinstance(k, int), f"Key '{k}' is not an integer."
        assert isinstance(v, list), f"Value of key '{k}' is not a list."

        for i, img in enumerate(v):
            assert isinstance(img, Image.Image), f"Element {i} of class {k} is not a PIL image."

    return data


def show_random_data(data, class2names, n=16):
    '''
    Show random images among data in a squared grid and save in data_summary/examples.png 
    '''
    assert m.sqrt(n) % 1 == 0, 'n_images must be a perfect square'
    grid_size = int(m.sqrt(n))
    all_images_with_labels = [(img, class_id) for class_id, images in data.items() for img in images]
    random_images = random.sample(all_images_with_labels, n)
    plt.figure(figsize=(15, 15))
    for i, (img, class_id) in enumerate(random_images):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img)
        plt.title(class2names[class_id], fontsize=15, weight='bold')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('data_summary/examples.png')


def show_frequencies(data, class2names):
    '''
    Show class frequencies of data file and save it in data_summary/class_frequencies.png
    '''
    # Counts images for each class (also ampty classes)
    class_counts = {class_id: len(data.get(class_id, [])) for class_id in class2names}
    total_images = sum(class_counts.values())

    # Relative frequencies computation
    class_freqs = {class_id: count / total_images if total_images > 0 else 0
                   for class_id, count in class_counts.items()}

    # Class id sorting
    sorted_class_ids = sorted(class2names.keys())

    # Equally distributing classes in groups (max 15 classes per group)
    num_groups = m.ceil(len(sorted_class_ids) / 15)  
    class_id_groups = np.array_split(sorted_class_ids, num_groups)

    # Max global frequency
    global_max_freq = max(class_freqs.values()) if total_images > 0 else 0.01
    ylim_top = max(global_max_freq * 1.15, 0.01)

    # Subplot setup
    ncols = min(num_groups, 2)
    nrows = m.ceil(num_groups / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.ravel(axes)  

    # Barplot creation
    for i, class_ids_group in enumerate(class_id_groups):
        ax = axes[i]

        labels = [class2names[cid] for cid in class_ids_group]
        freqs = [class_freqs[cid] for cid in class_ids_group]

        bars = ax.bar(range(len(labels)), freqs, color='steelblue')

        # Labels upon each bar
        for bar, freq in zip(bars, freqs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05*max(freqs),
                    f"{freq * 100:.2f}%", ha='center', va='bottom', fontsize=9)

        # Axis labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylim(0, ylim_top)
        ax.set_ylabel("Relative freq", weight='bold', size=15)

    # Hide extra subplots
    for j in range(len(class_id_groups), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle('Class frequencies', weight='bold', size=25)
    plt.tight_layout()
    plt.savefig('data_summary/class_frequencies.png')


def apply_gaussian_noise(img):
    '''
    Apply gaussian noise to an image. Noise will be zero-mean and random variance among 0.005, 0.01 and 0.02
    '''
    np_img = np.array(img)
    var = random.choice([0.005, 0.01, 0.02])
    gauss_img = random_noise(np_img, mode='gaussian', var = 0.005, clip=True)
    gauss_img = (gauss_img * 255).astype(np.uint8)
    return Image.fromarray(gauss_img)


def apply_sp_noise(img):
    '''
    Apply 5% salt&pepper noise to an image
    '''
    np_img = np.array(img)
    sp_img = random_noise(np_img, mode='s&p')
    sp_img = (sp_img * 255).astype(np.uint8)
    return Image.fromarray(sp_img)


def apply_blur_noise(img):
    '''
    Apply blur noise to an image. Blurring window size is randomly extracted among 3, 5 or 7
    '''
    k = random.choice([3, 5, 7])
    blurred_img = T.GaussianBlur(k)(img)
    return blurred_img


def apply_horizontal_flip(img):
    return F.hflip(img)


def apply_vertical_flip(img):
    return F.vflip(img)


def apply_random_rotation(img):
    '''
    Apply random rotation to an image. Rotation angle is randomly extracted from all multiples of 45°
    The filling colour is set as the median of pixel values for each channels
    '''
    # Median copmuting
    np_img = np.array(img)
    if len(np_img.shape) == 2:  
        fill = int(np.median(np_img))
    else:  
        fill = tuple(int(np.median(np_img[:, :, c])) for c in range(3))
    angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
    return F.rotate(img, angle, fill=fill, interpolation=InterpolationMode.BILINEAR)


class CustomDataset(Dataset):
    def __init__(self, data, transform=None, augmentation_p=None):
        super().__init__()
        self.samples = []
        self.transform = transform
        self.transformations = []
        self.p = []
        # list of supported augmentation types
        augmentation_map = {
            'gaussian': apply_gaussian_noise,
            's&p': apply_sp_noise,
            'blur': apply_blur_noise,
            'hflip': apply_horizontal_flip,
            'vflip': apply_vertical_flip,
            'rotate': apply_random_rotation
        }
        for aug, p in augmentation_p.items():
            if aug not in augmentation_map:
                raise ValueError(f"Augmentation '{aug}' not supported.")
            self.transformations.append(augmentation_map[aug])
            self.p.append(p)
        for class_label, elementi in data.items():
            for img in elementi:
                self.samples.append({
                    'image': img,
                    'label': int(class_label),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Apply all transformation specified in augmentation_p with the given probability. Then apply, if specified, the default transformations
        sample = self.samples[idx]
        image = sample['image']
        transformations_ids = [i for i in range(len(self.p))]
        np.random.shuffle(transformations_ids)
        for i in transformations_ids:
            if np.random.rand() < self.p[i]:
                image = self.transformations[i](image)
        if self.transform:
            image = self.transform(image)
        return image, sample['label']


def create_folds(data, n_folds, seed=42):
    '''
    Split the dataset into a given number of folds.
    Inputs:
    - data: data dict
    - n_folds: total number of desired folds
    - seed (optional): set for reproducibility

    Outputs:
    - folds: list of n_folds dicts. Every dict is in the same format of input data. Each fold keeps original class frequencies
    '''
    random.seed(seed)
    folds = [{} for _ in range(n_folds)]

    for class_id, images in data.items():
        tot_class_samples = len(images)
        indices = list(range(tot_class_samples))
        random.shuffle(indices)

        ceils = [True if i<tot_class_samples%n_folds else False for i in range(n_folds)]
        class_samples_per_fold = tot_class_samples / n_folds

        current_idx = 0
        for i in range(n_folds):
            n_fold_samples = m.ceil(class_samples_per_fold) if ceils[i] else int(class_samples_per_fold)
            fold_indices = indices[current_idx:current_idx+n_fold_samples]
            current_idx = current_idx+n_fold_samples

            folds[i][class_id] = [images[j] for j in fold_indices]
    return folds

def split_data(data, train_split, val_split, test_split, seed=42):
    '''
    Split data into train/val/test following the ratios given. 
    Class proportions are kept by random sampling 
    '''
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Proportions must sum to 1"

    random.seed(seed)

    train_data = {}
    val_data = {}
    test_data = {}

    for class_id, images in data.items():
        n = len(images)
        indices = list(range(n))
        random.shuffle(indices)

        n_train = int(train_split * n)
        n_val = int(val_split * n)
        n_test = n - n_train - n_val

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_data[class_id] = [images[i] for i in train_indices]
        val_data[class_id] = [images[i] for i in val_indices]
        test_data[class_id] = [images[i] for i in test_indices]

    return train_data, val_data, test_data


def create_dataloaders(data, train_config, data_config, mode='train', folds=None, fold_id=None, subfold_id=None, splits=None):
    '''
    Create train/val/test dataloaders.
    Inputs:
    - data: data dict 
    - train_config: train configuration
    - data_config: data configuration
    - folds: list of folds in which the data has been splitted
    - fold_id: current fold id
    - subfold_id: current subfold id
    - mode: ['train_val_test', 'train_test', 'train'] to specify the loaders to create
    - splits: {'train': 0.8, 'val': 0.1, 'test': 0.1} to specify fraction of data in each loader

    Split strategy: 
    - if mode = train, all data is converted in dataloader
    - if mode = train_test and data has already been splitted in folds, the fold indicated by the 'fold_id' param is converted in the test dataloader. 
      Remaining data will form the train dataloader. If there are no folds, split the dataset in train and test data
      following the ratios given by 'splits' param
    - if mode = train_val_test and data has already been splitted in folds, 'fold_id'-th fold will be the test dataloader, 
      the 'subfold_id'-th fold between the remaining folds will be the val dataloader, all the other folds will be merged in the train dataloader.
      If there are no folds, split the dataset in train and test data following the ratios given by 'splits' param
    '''
    # default transformation to be passed to the custom dataset
    transform_list = []
    if data_config['im_ch'] == 1:
        transform_list.append(T.Grayscale())
    transform_list += [
        T.Resize((data_config['im_size'], data_config['im_size'])),
        T.ToTensor()
    ]
    transform = T.Compose(transform_list)
    
    # Additional transformations will feature the train loader only
    train_aug_probs = train_config['augmentation_probs']
    test_aug_probs = {}

    # data split: train only
    if mode=='train':
        # all data will be converted in a single train loader
        train_data = data
        train_dataset = CustomDataset(train_data, transform=transform, augmentation_p=train_aug_probs)
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
        return train_loader

    # data split: train and test only
    elif mode=='train_test': 
        # if the dataset has already been splitted in folds
        if folds:
            test_data = folds[fold_id]
            merged = defaultdict(list)
            for fold in [folds[i] for i in range(train_config['n_folds']) if i != fold_id]:
                for class_id, images in fold.items():
                    merged[class_id].extend(images)
            train_data = dict(merged)
        # if there are no folds
        else:
            train_data, _, test_data = split_data(data, splits['train'], splits['val'], splits['test'], seed=random.randint(0, 1000))
        # loaders creation
        train_dataset = CustomDataset(train_data, transform=transform, augmentation_p=train_aug_probs)
        test_dataset = CustomDataset(test_data, transform=transform, augmentation_p=test_aug_probs)
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0,
                                  pin_memory=True, drop_last=False)
        return train_loader, test_loader

    # data split: train, val and test
    elif mode=='train_val_test':
        # if the dataset has already been splitted in folds
        if folds: 
            test_data = folds[fold_id]
            adj_subfold_id = subfold_id if subfold_id<fold else subfold_id+1
            val_data = folds[adj_subfold_id]
            merged = defaultdict(list)
            for fold in [folds[i] for i in range(train_config['n_folds']) if i not in [fold_id, adj_subfold_id]]:
                for class_id, images in fold.items():
                    merged[class_id].extend(images)
            train_data = dict(merged)
        # if there are no folds
        else:
            train_data, val_data, test_data = split_data(data, splits['train'], splits['val'], splits['test'], seed=random.randint(0, 1000))
        # loaders creation
        train_dataset = CustomDataset(train_data, transform=transform, augmentation_p=train_aug_probs)
        val_dataset = CustomDataset(val_data, transform=transform, augmentation_p=test_aug_probs)
        test_dataset = CustomDataset(test_data, transform=transform, augmentation_p=test_aug_probs)
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0,
                                      pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0,
                                  pin_memory=True, drop_last=False)
        return train_loader, val_loader, test_loader

    # if mode is other, raise error
    else:
        raise ValueError('mode not supported')