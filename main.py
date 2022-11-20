#!/home/yarinudi/DL_Project/venv/bin/python3.8
# -*- coding: utf-8 -*-
# %%
import sys
import os
import torch
import numpy as np

from data_handler import data_wrapper, train_test_split, load_extracted_features, process_feats
from dataset import  get_transforms, get_datasets, get_dataloaders
from train_seq_vit import evaluate_transformer, cross_valid_eval, cv_performance
from utils import seed_everything, get_cls_tokens, plot_auc_curves_per_frame
from seq_vit_utils import get_model_args, get_train_args, init_seq_vit
from intervention import get_cell_online_eval, plot_online_pred, intervene_frame, intervene_proba


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
    
    image_size, max_len = tot_feats[0].shape[1:], tot_feats[0].shape[1]

    train_args = get_train_args(epochs=100)

    # model_args = get_model_args(image_size, max_len, pad_token, device,
    #             dim=32, depth=9, heads=3, dim_head=16, mlp_dim=32)
    model_args = get_model_args(image_size, max_len, pad_token, device)

    # %% 
    """Split data to train and test. """
    train_list, test_list, labels_train, labels_test, test_idx = train_test_split(tot_feats, tot_labels, n_frames=1, on_raw_data=False)
    train_dataset, test_dataset = get_datasets(train_list, test_list, labels_train, labels_test, train_transforms, test_transforms)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, train_args, test_idx, verbose=1)
    
    # %%
    """ Evaluate a single model. """
    model = init_seq_vit(model_args, use_data_parallel=False)
    model, report = evaluate_transformer(model, train_loader, test_loader, train_args, model_args)

    # %%
    """10-folds cross-validation evaluation. """
    cv_results = cross_valid_eval(tot_feats, tot_labels, train_args, model_args, train_transforms=None, test_transforms=None, n_folds=10)

    # %%
    """10-folds cross-validation performance. """
    cv_perf = cv_performance(cv_results)

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
    """ Plot PRC AUC and ROC AUC curves on inference. """

    # train_list, test_list, labels_train, labels_test, test_idx = train_test_split(tot_feats, tot_labels, test_exps=test_idx, n_frames=1, on_raw_data=False)

    labels, probs = intervene_proba(model, test_list, labels_test, train_args, model_args, test_transforms)
    
    # plot_auc_curves_per_frame(labels, probs, frame=2, y_major=2)
    
    frames_to_explore = [2, 20, 50, 100, 166, 200, -1]
    [plot_auc_curves_per_frame(labels, probs, frame=frame, y_major=2) for frame in frames_to_explore]

# %%
