#!/home/yarinudi/DL_Project/venv/bin/python3.8
# -*- coding: utf-8 -*-
# %%
import sys
import os
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm 

from data_handler import data_wrapper, train_test_split, load_extracted_features, process_feats
from dataset import  get_transforms, get_datasets, get_dataloaders
from train_seq_vit import evaluate_transformer, cross_valid_eval, cv_performance
from utils import seed_everything, get_cls_tokens
from seq_vit_utils import get_model_args, get_train_args, init_seq_vit
from intervention import get_cell_online_eval, plot_online_pred, test_cell, intervene_frame


# %%
if __name__ == "__main__":
    # %%
    """load raw data + extracted features""" 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Working on: ', device)

    seed = 42
    seed_everything(seed)

    train_transforms, test_transforms = None, None

    raw_data_list, unique_labels = data_wrapper(get_train_list=False, get_labels=True)    
    tot_feats, tot_labels = load_extracted_features(unique_labels)

    # %%
    """ Process data: concatination, tokenization, padding. """
    
    SOS_token, EOS_token, pad_token = get_cls_tokens(num_feats=tot_feats[0].shape[1])

    tot_feats = process_feats(tot_feats, SOS_token, EOS_token, pad_token)
    
    image_size = tot_feats[0].shape[1:]

    train_args = get_train_args()

    # %% 
    """Split data to train and test. """
    train_list, test_list, labels_train, labels_test, test_idx = train_test_split(tot_feats, tot_labels, n_frames=1, on_raw_data=False)
    train_dataset, test_dataset = get_datasets(train_list, test_list, labels_train, labels_test, train_transforms, test_transforms)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, train_args, test_idx, verbose=1)
    
    # %%
    """ Evaluate a single model. """
    max_len = tot_feats[0].shape[1]

    model_args = get_model_args(image_size, max_len, pad_token)
    model = init_seq_vit(model_args)
    model, report = evaluate_transformer(model, train_loader, test_loader, train_args, model_args)

    # %%
    """10-folds cross-validation evaluation. """
    # cv_results = cross_valid_eval(n_folds=10)
    # cv_performance = cv_performance(cv_results)

    # %%
    """Local interpertability - specific cell predictions over time. """
    labels, preds, cell_idx =get_cell_online_eval(model, tot_feats, tot_labels, train_args, 
                                            model_args, test_transforms, max_len, cell_idx=None)
    plot_online_pred(labels, preds, np.array([*range(len(preds))]), cell_idx)

    # %%
    """Compute Metrics On Intervention Time"""
    # subsets - train test split
    train_list, test_list, labels_train, labels_test, test_idx = train_test_split(tot_feats, tot_labels, test_exps=test_idx, n_frames=1, on_raw_data=False)

    acc_signal, f1_signal, frame_idx = intervene_frame(model, test_list, labels_test, train_args, model_args, test_transforms)

# %%
