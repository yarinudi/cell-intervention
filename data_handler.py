import os
import random
import pickle
import utils
import numpy as np
import pandas as pd

from nd2reader import ND2Reader
import pims

from seq_vit_utils import pad_img


def get_map():
    """ 
    exp_map = get_map()
        
    Returns a dictionary with mapping of the video files to the correspending experiment.   
    """
    exp_map = dict({
        '1': ['02_03_2021_mp1_48_72h.nd2', '03_03_2021_mp1_72_94h.nd2'],
        '2': ['17012021_mp1_50hֹ70h_chir.nd2', '18012021_mp1_70h.nd2', '18012021_mp1_74h_95h.nd2'],
        '3': ['200521_mp1_exp8_48h_chir.nd2',  '200521_mp1_exp8_72h.nd2', '200521_mp1_exp8_87h_mgStart.nd2'],
        '4': ['202404_mp1_100c_48h_Chir.nd2', '202504_mp1_100c_72h_N2B27only.nd2'],
        '5': ['210202_exp6_mp1_100cells_48h.nd2', '210203_exp6_mp1_100cells_67h_sb.nd2', '210203_exp6_mp1_100cells_74h_SBconditions.nd2', 
            '210203_exp6_mp1_100cells_79h.nd2', '210204_exp6_mp1_100cells_94_97h.nd2']
    })

    return exp_map


def get_frames_and_attributes(f, type='nd2reader'):
    
    attributes = utils.get_attributes(f)
    sizes = f.sizes
    print("Video sizes: {0}".format(sizes))

    f.iter_axes, f.bundle_axes = 't', 'cyx'

    shape = (sizes['t'], sizes['v'], sizes['c'], sizes['y'], sizes['x']) if type=='nd2reader' else  (sizes['t'], sizes['m'], sizes['c'], sizes['y'], sizes['x'])
    frames = np.zeros(shape, dtype=np.int16)

    print('Loading frames...')
    for i in range(len(f)):
        frames[i] = f.get_frame(i)

    return frames, attributes


def save_to_data(frames, attributes, data, df_attributes):
    data.append(frames)
    df_attributes.loc[-1] = attributes
    df_attributes.index = df_attributes.index + 1

    return data, df_attributes


def load_labels(path):
    df_labels = pd.read_csv(path)
    df_labels.replace({'GBT': 2, 'GB': 1, 'other': 0, 'None': 0}, inplace=True)
    df_labels.set_index(df_labels.columns[0], inplace=True)
    df_labels = df_labels.T

    print('labels shape: ', df_labels.shape)
    # df_labels.hist(bins=3, legend=False, sharex=True, figsize=(20,15))

    return df_labels


def df_labels_to_list(df_labels, order):
    """convert labels dataframe to list of np arrays according to the order the videos were loaded. """
    
    order = list(map(lambda x: x.split('.')[0], order))
    labels = [df_labels[f'{col}'].values for col in order]
    
    return labels


def load_data(path):
    """ 
    data, df_attributes = load_data(path)
        
    Returns data of all nd2 files located in the input directory path.   

    Parameters
    ----------
        path : str - Files directory path. 

    Returns
    -------
        data : list - list of all the videos data in the input path.
        df_attributes: DataFrame - pandas dataframe with the attributes of each nd2 file. 
    """

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    df_attributes = pd.DataFrame(columns=['file_name', 'date', 'ndim', 'sizes', 'channels'])
    
    data = []
    files = sorted(os.listdir(path))
    files.remove('19042022_mp1_48h_ChirAll.nd2')
    files.remove('202204_mp1_74h_N2B27all.nd2')
    
    print("Loading files from {0}: ".format(files))

    for i, f_name in enumerate(files):

        print("Working on {0}".format(f_name))
        try:
            with ND2Reader(os.path.join(ROOT_DIR, path, f_name)) as f:
                frames, attributes = get_frames_and_attributes(f)
                # save img and attributes
                data, df_attributes = save_to_data(frames, attributes, data, df_attributes)

        except TypeError as e:
            with pims.ND2Reader_SDK(os.path.join(ROOT_DIR, path, f_name)) as f:
                frames, attributes = get_frames_and_attributes(f, type='pims')
                # save img and attributes
                data, df_attributes = save_to_data(frames, attributes, data, df_attributes)
        
    df_attributes.reset_index(drop=True, inplace=True)
    print('Finished loading {0} videos !'.format(len(data)))
    print(df_attributes)

    return data, df_attributes


def concatenate_vids_of_same_exp(data, labels, data_order):
    exp_map = get_map()
    n_cells_in_vid = 60
    train_list, labels_list = [], []
    
    for key, vals in exp_map.items():
        data_idx = np.array([i for i, val in enumerate(data_order) if val in vals])
        print(f'working on experiment {key}... ')

        for cell in range(n_cells_in_vid):

            temp_data, temp_labels = [], []

            for i in data_idx:
                temp_data.append(data[i][:, cell, :, :, :])
                temp_labels.append(labels[i][cell])
            
            train_list.append(temp_data)
            labels_list.append(temp_labels)

    return train_list, labels_list


def get_loaded_data_order(path='resources/data/raw_data/'):
    data_order = sorted(os.listdir(path))
    data_order.remove('19042022_mp1_48h_ChirAll.nd2')
    data_order.remove('202204_mp1_74h_N2B27all.nd2')

    return data_order


def train_test_split(feats, labels, n_frames=1, test_exps=None, on_raw_data=True, high=15, size=3):
    
    n_cells = 60
    test_exps = random.sample(range(1, high+1), size) if test_exps is None else test_exps
    
    test_idxs = [i for idx in test_exps for i in [*range(n_cells*(idx - 1)*n_frames, n_cells*(idx)*n_frames)]] if not on_raw_data else [i for idx in test_exps for i in [*range(n_cells*(idx - 1), n_cells*idx)]]
    train_idxs = [x for x in range(len(feats)) if x not in test_idxs]
    train_list, test_list = [feats[i] for i in train_idxs], [feats[i] for i in test_idxs]
    labels_train, labels_test = [labels[i] for i in train_idxs], [labels[i] for i in test_idxs]

    print(f"Test experiments are: {test_exps}. \n Train Data: {len(train_list)} \n Test Data: {len(test_list)}")

    return train_list, test_list, labels_train, labels_test, test_exps
    

def data_wrapper(get_train_list=False, get_labels=False):
    
    data_path, labels_path = 'resources/data/raw_data/', 'resources/data/labels.csv'

    if not get_train_list and get_labels:
        print('loading labels..')
        data_order = get_loaded_data_order(data_path)
    
        df_labels = load_labels(labels_path)
        labels = df_labels_to_list(df_labels, order=data_order)

        # concatenate labels of the same experiment
        exp_map = get_map()
        n_cells_in_vid = 60
        labels_list = []
        
        for vals in exp_map.values():
            data_idx = np.array([i for i, val in enumerate(data_order) if val in vals])

            for cell in range(n_cells_in_vid):
                temp_labels = []

                temp_labels = [labels[i][cell] for i in data_idx]
                labels_list.append(temp_labels)


        unique_labels = [int(np.unique(x)) for x in labels_list]
        # unique_labels = [1 if x == 2 else 0 for x in unique_labels] # convert to binary classification
        
        return None, unique_labels

    raw_data, df_attributes = load_data(data_path)
    df_labels = load_labels(labels_path)

    data_order=df_attributes['file_name'].values
    labels = df_labels_to_list(df_labels, order=data_order)

    # concatenate videos of the same experiment
    train_list, labels_list = concatenate_vids_of_same_exp(raw_data, labels, data_order)
    
    unique_labels = [int(np.unique(x)) for x in labels_list]
    # unique_labels = [1 if x == 2 else 0 for x in unique_labels] # convert to binary classification
    
    return train_list, unique_labels


def load_extracted_features(unique_labels):
    feats = []
    feats_path = os.path.join(os.getcwd(), 'resources/data/densenet_feats')
    files = os.listdir(feats_path)
    
    for f in files:
        print('loading ', f)
        with open(os.path.join(feats_path, f), 'rb') as handle:
            temp_feats = pickle.load(handle)
            feats.append(temp_feats)

    feats = [*feats[0], *feats[1], *feats[2]]
    labels = [*unique_labels, *unique_labels, *unique_labels]
    
    print(len(feats))
    print('Extracted features were loaded successfully. \n ******************************************')

    return feats, labels


def process_feats(feats, SOS_token, EOS_token, pad_token):
    """ 
    feats = process_feats(feats, SOS_token, EOS_token, pad_token)
        
    Returns features list in the structure needed for training a SeqViT model.     
    
    Parameters
    ----------
        feats : list [array] - total features list. 
        SOS_token : array [float] - start of sequence token.
        EOS_token : array [float] - end of sequence token.
        pad_token : array [float] - padding token.

    Returns
    -------
        feats : list [array] - processed total features list. 

    """
    max_len = max([len(feat) for feat in feats]) + 2

    feats = [np.concatenate((SOS_token, feat, EOS_token), axis=0) for feat in feats]
    feats = [pad_img(feat, max_len, pad_token) for feat in feats]
    feats = [feat.reshape((1, max_len, -1)) for feat in feats]

    return feats