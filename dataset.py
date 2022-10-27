import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)


def get_transforms(feature_extractor=None, reshape_size=256):
    """ Returns transforms for train and test datasets. """
    
    if feature_extractor is None:
            
        train_transforms = Compose([
                    RandomResizedCrop(reshape_size),
                    RandomHorizontalFlip(),
                ]
            )

        test_transforms = Compose([
                    Resize(reshape_size),
                    CenterCrop(reshape_size),
                ]
            )

        return train_transforms, test_transforms

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose([
                RandomResizedCrop(feature_extractor.size),
                RandomHorizontalFlip(),
                # ToTensor(),
                normalize,
            ]
        )

    test_transforms = Compose([
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                # ToTensor(),
                normalize,
            ]
        )

    return train_transforms, test_transforms


def plot_dataset_traget_histograms(train_dataset, test_dataset, test_idx):
    """ Displays datasets' target histograms. """

    bins = len(np.unique(train_dataset.get_labels()))
    plt.figure(figsize=(15, 8))
    plt.subplot(211)
    plt.hist(train_dataset.get_labels(), bins=bins)
    plt.title('Train Dataset - Targets Histogram')
    plt.subplot(212)
    plt.hist(test_dataset.get_labels(), bins=bins)
    plt.title(f'Test Dataset (Exp. {test_idx}) - Targets Histogram')


def get_datasets(train_list, test_list, labels_train, labels_test, train_transforms, test_transforms):
    """ Returns train and test CellDataset datasets objects. """

    train_dataset = CellsDataset(train_list, labels_train, transform=train_transforms)
    test_dataset = CellsDataset(test_list, labels_test, transform=test_transforms)

    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, train_args, test_idx=None, verbose=0):
    """ Returns train and test DataLoaders with imbalanced data sampler. """

    train_loader = DataLoader(dataset=train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), \
                        batch_size=train_args['batch_size'])

    test_loader = DataLoader(dataset=test_dataset, batch_size=train_args['batch_size'], shuffle=False)


    print(len(train_dataset), len(train_loader))
    print(len(test_dataset), len(test_loader))
    
    if verbose:
        plot_dataset_traget_histograms(train_dataset, test_dataset, test_idx)

    return train_loader, test_loader


class CellsDataset(Dataset):
    """ 
    dataset = CellsDataset(feats_list, labels, transform)
        
    Returns a CellsDataset dataset.
    
    Parameters
    ----------
        feats_list : list [array] - list of np arrays with the videos features.
        labels : list [int] - list of labels with the same length as the features.
        transform : transforms - torchvision transforms to apply on the dataset.

    Returns
    -------
        dataset : CellsDataset - a dataset. 

    """
    def __init__(self, feats_list, labels, transform=None):
        self.feats_list = feats_list
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.feats_list)
        return self.filelength

    def __getitem__(self, idx):
        img = self.feats_list[idx]
        img = torch.from_numpy(np.array(img).astype(np.float32))
        img_transformed = self.transform(img) if self.transform is not None else img 
        label = self.labels[idx]

        return img_transformed, label

    def get_labels(self):
        return self.labels
