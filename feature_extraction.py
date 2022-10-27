import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator


def img_aug(img, n_aug_imgs=1, batch_size=1):
    """ 
    aug_imgs = img_aug(img, n_aug_imgs=1, batch_size=1)
        
    Returns augmentations for a specific frame.
    
    Parameters
    ----------
        img : array - specific frame to be augmented. 
        n_aug_imgs : int - number of augmentations to perform.
        batch_size : int - number of frames to be augmented (currently, the recommendation is 1). 
    
    Returns
    -------
        aug_imgs : array - frame augmentations. 

    """
   
    datagen = ImageDataGenerator(width_shift_range=0.3, 
                                height_shift_range=0.3,
                                vertical_flip=True, 
                                horizontal_flip=True,
                                rotation_range=180,
                                brightness_range=[0.7, 1.3],
                                zoom_range=0.3)

    it = datagen.flow(img, batch_size=batch_size, seed=42)
    # aug_imgs = [it.next()[0] for i in range(n_aug_imgs)]
    aug_imgs = it.next()[0]
    aug_imgs = np.reshape(aug_imgs, (1, aug_imgs.shape[0], aug_imgs.shape[1], aug_imgs.shape[2]))

    return aug_imgs


def subset_aug(subset):
    """ 
    aug_imgs = subset_aug(subset)
        
    Returns augmentations to input subset frames.
    
    Parameters
    ----------
        subset : list [array] - input list of frames. 

    Returns
    -------
        aug_imgs : list [array] - augmented frames. 

    """
    ex_img = subset[0]
    h, w, ch = ex_img.shape[2], ex_img.shape[1], ex_img.shape[0] 

    subset = [np.reshape(frame.T, (1, h, w, ch)) for frame in subset]

    aug_imgs = [img_aug(img) for img in subset]

    return aug_imgs
    

def init_aug(train_list):
    """ 
    init_aug(train_list)
        
    Executing image augmentations.
    
    Parameters
    ----------
        train_list : list [array] - input list of frames.  

    Returns
    -------
        None. 

    """

    for i in range(len(train_list)):
        print(f'working on cell no. {i} ...')
        aug_imgs = [subset_aug(subset) for exp in train_list for subset in exp]
        
        # save aug imgs
        with open(os.path.join(os.getcwd(), f'resources/data/aug_imgs/aug_imgs_cell_{i}.pickle'), 'wb') as handle:
            pickle.dump(aug_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Finished augmenting images !')


def extract_feats(X, model, n_batches):
    """ 
    feats = extract_feats(X, model, n_batches)
        
    Returns .
    
    Parameters
    ----------
        X : list [array] - raw frames data. 
        model : any model with predict method - feature extractor. 
        n_batches : int - number of batches to predict.

    Returns
    -------
        feats : array - model extracted features. 

    """

    feats = []

    for i in range(n_batches):
        img = preprocess_input(X[i])
        preds = model.predict(img, verbose=0)

        feats.append(preds.flatten())

    feats = np.array(feats)
    print('finished extracting features. \n features shape: ', feats.shape)
    
    return feats


def init_feature_extractor_model(img_size):
    """ 
    model = init_feature_extractor_model(img_size)
        
    Returns a DenseNet121 feature extractor model.
    
    Parameters
    ----------
        img_size : tuple [ints] - model's input image. 

    Returns
    -------
        model : keras model - densenet121 feature extractor model. 

    """

    inp = Input(img_size)
    backbone = DenseNet121(input_tensor=inp, include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:,:,0])(x)
    
    model = Model(inp, out)
    # print(model.summary())
    # plot_model(model, to_file='/home/yarinudi/DL_Project/resources/model_plot.png', show_shapes=True, show_layer_names=True)

    return model


def extract_features(train_list, extract_from_aug=False):
    """ 
    model = extract_features(img_size)
        
    Extracting features from raw data using DenseNet12 model and saving it to resources path.
    
    Parameters
    ----------
        train_list : list [array] - features to be augmented. 
        extract_from_aug : bool - flag to mark if raw data should be augmented.

    Returns
    -------
        None.

    """

    feats = []

    for i, cell_vid in enumerate(train_list):

        print(f'extracting feature from cell {i}...')

        cell_feats = []
        for cell_subset in cell_vid:
        
            ex_img = cell_subset[0]
            h, w, ch = ex_img.shape[2], ex_img.shape[1], ex_img.shape[0] 
            img_size = (h, w, ch)
            model = init_feature_extractor_model(img_size)
            n_batches = cell_subset.shape[0]
            
            X = [np.reshape(frame.T, (1, h, w, ch)) for frame in cell_subset]

            if extract_from_aug:
                aug_imgs = [img_aug(img) for img in X]
                
            temp_extract_feats = extract_feats(X, model, n_batches) if not extract_from_aug else extract_feats(aug_imgs, model, n_batches)
            cell_feats.append(temp_extract_feats)

        feats.append(np.concatenate(cell_feats))
        print(f'finished extracting new features for cell {i}.\n  ----------------')

    # save features
    f_name = 'extracted_feats.pickle' if not extract_from_aug else 'extracted_aug_imgs_feats_1.pickle'

    with open(f_name, 'wb') as handle:
        pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Feature extraction finished with {0} new features.'.format(len(feats)))

    return feats


if __name__ == "__main__":
    '''Run this code in order to augment the raw data (e.g. videos) and then extract features with Denset-121.'''
    
    print(f"Torch: {torch.__version__}")
    print("is cuda available? ", torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 42
    seed_everything(seed)

    train_transforms, test_transforms = None, None

    # load raw data 
    raw_data_list, unique_labels = data_wrapper(get_train_list=False, get_labels=True)    

    # raw data augmentations
    init_aug(train_list)
    feats = extract_features(train_list, extract_from_aug=True)
