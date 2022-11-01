import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm 

from dataset import CellsDataset
from train_seq_vit import test, predict_proba
from seq_vit_utils import pad_img


def test_cell(model, cell_feats, cell_labels, train_args, model_args, test_transforms, get_proba=False):            
    """ 
    labels, preds = test_cell(model, cell_feats, cell_labels, train_args, model_args, test_transforms, max_len)
    
    Returns labels and predictions of specific cell.
    
    Parameters
    ----------
        model : Model - the model to use to test sample.
        cell_feats : array [float] - a single sample features.
        cell_labels : array [int] - a single sample labels. 
        train_args : dict - contains the ttaining arguments.
        model_args : dict - contains the model arguments.
        test_transforms : transforms - torchvision transforms to apply on the dataset.
        max_len : int - the maximum length a sample can be. 

    Returns
    -------
        labels : list [int] - labels of input features.
        preds : list [int] - predictions of input features. 

    """
    max_len = model_args['max_len']

    data = [cell_feats[:, :i + 1, :] for i in range(cell_feats.shape[1])]
    data = [pad_img(feats.squeeze(0), max_len, model_args['pad_token']) for feats in data]
    data = [feats.reshape((1, max_len, -1)) for feats in data]
    cell_labels = np.repeat(cell_labels, len(data))

    cell_dataset = CellsDataset(data, cell_labels, transform=test_transforms)
    cell_loader = DataLoader(dataset=cell_dataset, batch_size=train_args['batch_size'], shuffle=False)

    labels, preds = test(model, cell_loader, model_args, load_from_dir=False) if not get_proba else \
        predict_proba(model, cell_loader, model_args, load_from_dir=False)

    return labels, preds


# local interpretability
def get_cell_online_eval(model, feats, labels, train_args, model_args, test_transforms, max_len, cell_idx=None):
    """ 
    labels, preds, cell_idx = get_cell_online_eval(model, feats, labels, train_args, test_transforms, device, cell_idx)
        
    Returns data relevant to plot cell's predictions over time.
    
    Parameters
    ----------
        model : Model - model to use to get predictions.
        feats : list [arrayas] - total dataset features.
        labels : list [int] - total dataset labels.
        train_args : dict - dictionary contains all the training arguments.
        test_transforms : transforms - torchvision transforms to apply on the dataset.
        device : str - the device to use to predict.
        cell_idx : int - sample's index to get data for.

    Returns
    -------
        labels : list - predictions of input features. 
        preds : list - predictions of input features. 
        cell_idx : int - index of the sample to plot. 
    """
    
    cell_idx = np.random.randint(low=0, high=len(feats)) if cell_idx is None else cell_idx
    cell_feats, cell_labels = feats[cell_idx], labels[cell_idx]

    labels, preds = test_cell(model, cell_feats, cell_labels, train_args, model_args, test_transforms)

    return labels, preds, cell_idx


def plot_online_pred(labels, preds, t, idx):
    """ 
    plot_online_pred(labels, preds, t, idx)
        
    Display figure with predictions over time for a single cell.
    
    Parameters
    ----------
        labels : list [int] - list of labels. 
        preds : list [int] - list of predictions.
        t : list [int] - frames' time range.
        idx : int - cell index in the dataset.

    Returns
    -------
        None. 

    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=t, y=(labels == preds)*1, name="Correct"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=t, y=preds, name="Prediction"),
        secondary_y=False
    )


    fig.update_layout(
        title_text=f"Online Prediction - Exp. #{int(np.ceil(idx/60))} | Cell #{int(idx)} | Label {np.unique(labels)[0]}",
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="# Frame")
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_yaxes(title_text="<b>Correct</b>", secondary_y=True)
    fig.update_yaxes(title_text="<b>Prediction</b>", secondary_y=False)

    fig.show()


def intervene_frame(model, test_list, labels_test, train_args, model_args, test_transforms):
    """Returns the best time to intervene."""
    
    print('computing best intervention time...')
    labels, preds = [], []

    for cell_idx in tqdm(range(len(test_list))):
        cell_labels, cell_preds = test_cell(model, test_list[cell_idx], labels_test[cell_idx], train_args, model_args, test_transforms)
        labels.append(cell_labels), preds.append(cell_preds)

    temp_preds = np.array(preds).T
    temp_labels = np.array(labels).T

    acc_signal = list(map(lambda i: accuracy_score(temp_labels[i], temp_preds[i]), range(len(temp_preds))))
    f1_signal = list(map(lambda i: f1_score(temp_labels[i], temp_preds[i], average='weighted'), range(len(temp_preds))))
    frame_idx = np.argmax(f1_signal)
    print(f'The best frame to intervene is: {frame_idx} \n and the best score is {f1_signal[frame_idx]}')
    plot_intervention_scores(f1_signal, acc_signal, test_list)
    
    return acc_signal, f1_signal, frame_idx


def intervene_proba(model, test_list, labels_test, train_args, model_args, test_transforms):
    """Returns the probabilities of each frame."""
    
    print('computing best intervention time...')
    labels, probs = [], []

    for cell_idx in tqdm(range(len(test_list))):
        cell_labels, cell_probs = test_cell(model, test_list[cell_idx], labels_test[cell_idx], train_args, model_args, test_transforms, get_proba=True)
        labels.append(cell_labels), probs.append(cell_probs)

    return np.array(labels), np.array(probs)


def plot_intervention_scores(f1_signal, acc_signal, test_list):
    """ Displays F1 signal and accuracy signals attained from cross-validation evalutaion. """
    
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    ax1, ax2 = axes.flatten()

    ax1.plot(f1_signal, 'r')
    ax1.set(ylabel='Score', title='F1 Score (weighted)')

    ax2.plot(acc_signal, 'b')
    ax2.set(xlabel='Frame', ylabel='Score', title='Accuracy')

    plt.suptitle(f'Test Dataset - {len(test_list)} Samples')
    plt.tight_layout()
    plt.show()
