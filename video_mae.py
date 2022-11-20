# %%
import sys
import os
import pickle

import torch

from tqdm import tqdm

import numpy as np

from transformers import VideoMAEFeatureExtractor

from data_handler import data_wrapper
from feature_extraction import img_aug
from utils import seed_everything

"""
    This job's task to extract features out of a video using VideoMAE feature extractor. 
"""

# %%
if __name__ == "__main__":
    """load raw data + extracted features""" 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Working on: ', device)

    seed = 42
    seed_everything(seed)

    # %%
    """ VideoMAE feature extractor. """

    def feature_extractor_video_mae(videos, is_aug_vids=False, n_augs=2, idx=None):    
        """ 
        extract_feats_video_mae(videos, is_aug_vids)
            
        Extracting features from raw data using VideoMAE feature extractor and saving it to resources path.
        
        Parameters
        ----------
            data : list [array] - videos to be augmented. 
            is_aug_vids : bool - flag to mark if raw data should be augmented.

        Returns
        -------
            None.
        """

        feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
        vid_idx = 0 if idx is None else idx

        for vid in tqdm(videos):
            
            vid = [item for sublist in vid for item in sublist]
            vid_feats, aug_vids = [], []

            if is_aug_vids:

                for i in range(n_augs):
                    aug_vid = []
                    for frame in vid:

                        h, w, ch = frame.shape[2], frame.shape[1], frame.shape[0] 
                        frame =  np.reshape(frame, ((1, h, w, ch)))  
                        aug_frame = img_aug(frame)
                        aug_vid.append(torch.Tensor(aug_frame.T).squeeze())

                    aug_vids.append(aug_vid)
            
            vid = [torch.Tensor(frame) for frame in vid]

            temp_extract_feats = feature_extractor(vid, return_tensors="pt").pixel_values
            vid_feats.append(temp_extract_feats)

            for temp_vid in np.array(aug_vids):
                temp_extract_feats = feature_extractor(list(temp_vid), return_tensors="pt").pixel_values
                vid_feats.append(temp_extract_feats)

            # save features
            f_name = f'videomae_feats_{vid_idx}.pickle'

            with open(os.path.join('resources/data/videomae_feats', f_name), 'wb') as handle:
                pickle.dump(vid_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

            vid_idx += 1

    # raw_data_list, _ = data_wrapper(get_train_list=True, get_labels=False)
    # feature_extractor_video_mae(raw_data_list, is_aug_vids=True)
    # print('Done')
    
    # %%
    """ load extracted videos. """

    def load_videos_mae_feats(path):
        saving_path = os.path.join(os.getcwd(), path)
        file_paths = [os.path.join(saving_path, f_name) for f_name in os.listdir(saving_path)]

        feats = []
        for f in tqdm(file_paths):
            with open(f, 'rb') as handle:
                temp = pickle.load(handle)
                feats.append(temp)

        return feats

    path = 'resources/data/videomae_feats'
    feats = load_videos_mae_feats(path)

# %%
