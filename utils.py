import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def seed_everything(seed):
    """ Seeds all relevant random generators to the same value. """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_attributes(f):
    """ 
    attributes = get_attributes(f)
        
    Returns the attributes of the input nd2 file.   

    Parameters
    ----------
    f : str - The opened nd2 file 

    Returns
    -------
    attributes : list - a list of strings represents the attributes of the file. 

    """

    try:    
        attributes = [
            f.filename.split('/')[-1], f.metadata['date'], 
            f.ndim, f.sizes, f.metadata['channels']
        ]

    except Exception as e:
        attributes = [
            f.filename.split('/')[-1], f.filename.split('/')[-1].split('_')[0], 
            5, f.sizes, ('Ph', 'GFP', 'RFP(G)')
        ]

    return attributes


def plot_channels_with_hist(x, title):
    """ Displays a figure with the channels of an image and their histogram. """
    
    fig = plt.figure(figsize=(8, 10))
    plt.subplot(241); plt.imshow(x[0] + x[1] + x[2]); plt.title('All Channels')
    plt.subplot(242); plt.imshow(x[0]); plt.title('Channel Ph')
    plt.subplot(243); plt.imshow(x[1]); plt.title('Channel GFP')
    plt.subplot(244); plt.imshow(x[2]); plt.title('Channel RFP')

    plt.subplot(245); plt.hist(x[0] + x[1] + x[2], bins=10)
    plt.subplot(246); plt.hist(x[0], bins=10)
    plt.subplot(247); plt.hist(x[1], bins=10)
    plt.subplot(248); plt.hist(x[2], bins=10)
    

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def verify_data(data):
    """" Prints data shapes for verification. """

    print(data[0].shape) # video no. (111, 60, 3, 520, 696)
    print(data[0][8].shape) # timestamp no. (60, 3, 520, 696)
    print(data[0][8][42].shape) # cell no. (3, 520, 696)
    print(data[0][8][42][1].shape) # channel no. 1 (520, 696)
    print(data[0][8][42][1]) # frame XxY


def plot_random_img_from_data(data):
    """ 
    plot_random_img_from_data(data)
        
    Displays a random image from the data.
    
    Parameters
    ----------
        data : list [array] - list of videos. 

    Returns
    -------
        None.

    """
    train_list = data[-1]
    random_idx = np.random.randint(1, len(train_list))

    fig, axes = plt.subplots(1, 3, figsize=(16, 12))

    for idx, ax in enumerate(axes.ravel()):
        img = Image.fromarray(
            ((train_list[random_idx][idx] - np.min(train_list[random_idx][idx]))* 255)/(np.max(train_list[random_idx][idx]) - np.min(train_list[random_idx][idx])).astype(np.uint8)
            )
        ax.set_title(idx)
        ax.imshow(img)


def get_cls_tokens(num_feats):
    """ Returns classification tokens to be used. """

    SOS_token = nn.Parameter(torch.randn(1, num_feats)).detach().numpy()
    EOS_token = nn.Parameter(torch.randn(1, num_feats)).detach().numpy()
    pad_token = nn.Parameter(torch.randn(1, num_feats)).detach().numpy().T

    return SOS_token, EOS_token, pad_token