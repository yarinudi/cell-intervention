import os
from unittest.mock import patch
import torch
import torch.nn as nn
import numpy as np

from seq_vit import SeqViT


def get_train_args(batch_size=128, epochs=250, lr=0.001, weight_decay=0.1, 
                    gamma=0.9, focal_loss_alpha=1.2, focal_loss_gamma=3, mpp_loss_alpha=1.8):
    """ 
    train_args = get_train_args(args)
        
    Returns a dictionary with input arguments to train a model by.
    
    """
    train_args = {
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'gamma': gamma,
        'focal_loss_alpha': focal_loss_alpha,
        'focal_loss_gamma': focal_loss_gamma,
        'mpp_loss_alpha': mpp_loss_alpha
    }

    return train_args


def get_model_args(image_size, max_len, pad_token, device='cuda', patch_size=(1, 256), num_classes=3,
                    max_pixel_val=5.96, dim=64, depth=7,
                    heads=3, mlp_dim=128, channels=1, dim_head=32, dropout_p=0):
    """ 
    model_args = get_model_args(args)
        
    Returns a dictionary with input arguments to build a model by.
    
    """

    model_args =  {
        'image_size': image_size,
        'max_len': max_len,
        'patch_size': patch_size,
        'num_classes': num_classes,
        'device': device,
        'max_pixel_val': max_pixel_val,
        'dim': dim,
        'depth': depth,
        'heads': heads,
        'mlp_dim': mlp_dim,
        'channels': channels,
        'dim_head': dim_head,
        'dropout_p': dropout_p,
        'pad_token': pad_token
    }

    return model_args


def init_seq_vit(model_args, use_data_parallel=False):
    """ 
    model = init_seq_vit(model_args, use_data_parallel=False)
        
    Returns SeqViT model with the input arguments.
    
    Parameters
    ----------
        model_args : dict - contains model arguments. 
        use_data_parallel : bool - flag to use multiple devices.

    Returns
    -------
        model : SeqViT - sequence vision transformer model . 

    """
    model = SeqViT(
        image_size=model_args['image_size'],
        patch_size=model_args['patch_size'],
        num_classes=model_args['num_classes'],
        dim=model_args['dim'],
        depth=model_args['depth'],
        heads=model_args['heads'],
        mlp_dim=model_args['mlp_dim'],
        channels=model_args['channels'],
        dim_head=model_args['dim_head'],
        dropout_p=model_args['dropout_p'],
        max_len=model_args['max_len'],
        device=model_args['device'],
        max_pixel_val=model_args['max_pixel_val']
    )

    if use_data_parallel and torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(model_args['device'])

    return model


def pad_img(img, max_len, pad_token):
    """ 
    img = pad_img(img, max_len, pad_token)
        
    Returns padded image according to maximum length.
    
    Parameters
    ----------
        img : array [float] - input image to be padded. 
        max_len : int - maximum length of series in the dataset.
        pad_token : tensor - feature vector who chosen to be the padding token.

    Returns
    -------
        img : array [float] - padded image. 

    """
    seq_len = img.shape[0]

    if seq_len < max_len:
        padding = pad_token*np.ones((1, (max_len - seq_len)))
        img = np.concatenate([img, padding.T])

    return img
