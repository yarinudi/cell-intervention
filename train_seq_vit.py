import os
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


from data_handler import data_wrapper, train_test_split, load_extracted_features
from dataset import  get_datasets, get_dataloaders
from mpp import init_mpp
from focal_loss import FocalLoss
from seq_vit_utils import init_seq_vit


def evaluate_transformer(model, train_loader, val_loader, train_args, model_args, mae=None):

    train_acc_tot, val_acc_tot = [], []
    train_loss_tot, val_loss_tot = [], []

    best_model = copy.deepcopy(model.state_dict())
    best_acc, best_loss = 0.0, np.Inf

    criterion = FocalLoss(alpha=train_args['focal_loss_alpha'], gamma=train_args['focal_loss_gamma'])
    alpha = train_args['mpp_loss_alpha']

    optimizer = optim.AdamW(model.parameters(), lr=train_args['lr'])
    scheduler = StepLR(optimizer, step_size=3, gamma=train_args['gamma'])

    for epoch in range(train_args['epochs']):
        epoch_loss = 0
        epoch_accuracy = 0
        
        model.train()
        for data, label in tqdm(train_loader):
            mpp_trainer = init_mpp(model, data, model_args)
            data, orig_img, mask = mpp_trainer.mask_input(data)

            data, label, orig_img, mask = data.to(model_args['device']), label.to(model_args['device']), orig_img.to(model_args['device']), mask.to(model_args['device'])
            output, mpp_loss = model(data, orig_img, mask)
            # loss = criterion(output, label) if not mae else mae(data)
            loss = alpha*mpp_loss + criterion(output, label)

            if epoch != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            torch.cuda.empty_cache()

        train_acc_tot.append(epoch_accuracy.detach().cpu().numpy())
        train_loss_tot.append(epoch_loss.detach().cpu().numpy())

        if epoch > 3 and train_loss_tot[-1] > train_loss_tot[-2]:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0

            for data, label in tqdm(val_loader):
                mpp_trainer = init_mpp(model, data, model_args)
                data, orig_img, mask = mpp_trainer.mask_input(data)

                data, label, orig_img, mask = data.to(model_args['device']), label.to(model_args['device']), orig_img.to(model_args['device']), mask.to(model_args['device'])

                val_output, mpp_loss = model(data, orig_img, mask)
                # val_loss = criterion(val_output, label)
                val_loss = alpha*mpp_loss + criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)
                torch.cuda.empty_cache()

        val_acc_tot.append(epoch_val_accuracy.detach().cpu().numpy())
        val_loss_tot.append(epoch_val_loss.detach().cpu().numpy())

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = copy.deepcopy(model.state_dict())

            print('new model was saved successfully.')

    model.load_state_dict(best_model)

    # Save Model
    torch.save(best_model, os.path.join(os.getcwd(), 'resources', 'model', 'best_model.pth'))

    # display performance for the best model
    _ = plot_performace(model, train_loader, model_args, title='Train Dataset')
    report = plot_performace(model, val_loader, model_args)

    plot_model_accuracy_and_loss(train_loss_tot, val_loss_tot, train_acc_tot, val_acc_tot)

    print('Done!')
    return model, report


def test(model, test_loader, model_args, load_from_dir=False):
    labels, preds = [], []

    if load_from_dir:
        model_path = os.path.join(os.getcwd(), 'resources', 'model', 'best_model.pth')
        model = init_seq_vit(model_args)
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            mpp_trainer = init_mpp(model, data, model_args)
            data, orig_img, mask = mpp_trainer.mask_input(data)

            data, label, orig_img, mask = data.to(model_args['device']), label.to(model_args['device']), \
                                    orig_img.to(model_args['device']), mask.to(model_args['device'])

            pred, _ = model(data, orig_img, mask)
            pred = pred.argmax(dim=1)

            labels.append(label.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

            torch.cuda.empty_cache()

    labels = [label for curr_labels in labels for label in curr_labels] 
    preds = [pred for curr_preds in preds for pred in curr_preds] 
    
    return np.array(labels), np.array(preds)


def cross_valid_eval(feats, labels, train_args, model_args, train_transforms=None, test_transforms=None, n_folds=10):

    cv_results = {
        'best_models': [],
        'classification_report': []
        }

    for i in range(n_folds):
        print('Vlidation ', i)
        # subsets - train test split
        train_list, test_list, labels_train, labels_test, test_idx = train_test_split(feats, labels, n_frames=1, on_raw_data=False)

        train_dataset, test_dataset = get_datasets(train_list, test_list, labels_train, labels_test, train_transforms, test_transforms)
        train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, train_args, test_idx=None, verbose=1)

        model = init_seq_vit(model_args)
        model, report = evaluate_transformer(model, train_loader, test_loader, train_args, model_args)

        cv_results['best_models'].append(model)
        cv_results['classification_report'].append(report)
    
    print('Cross Validation - Done')

    return cv_results


def get_metrics_per_target(label):
    precision_list = np.array([rep[label][-1]['precision'] for rep in report_tot])
    recall_list = np.array([rep[label][-1]['recall'] for rep in report_tot]) 
    f1_list = np.array([rep[label][-1]['f1-score'] for rep in report_tot]) 

    print(f'Evaluation Metrics for Label {label}: \n Precision - {precision_list.mean()} +/- {precision_list.std()}' \
            f'\n Recall - {recall_list.mean()} +/- {recall_list.std()}' \
            f'\n F1-Score - {f1_list.mean()} +/- {f1_list.std()}')

    return precision_list, recall_list, f1_list


def cv_performance(cv_results):
    """Calculates 10 fold metrics. """

    report_tot = [[item[-3:] for item in rep.items()] for rep in cv_results['classification_report']]
    accuracy_list = np.array([rep[-3][-1] for rep in report_tot])
    f1_list = np.array([rep[-1][-1]['f1-score'] for rep in report_tot])

    mean_accuracy, std_accuracy = accuracy_list.mean(), accuracy_list.std()
    mean_weighted_f1_score, std_f1_score =f1_list.mean(), f1_list.std()
    print(f"Total Mean Accuracy: {mean_accuracy} +/- {std_accuracy} \n Total Mean Weighted F1-Score {mean_weighted_f1_score} +/- {std_f1_score}")

    cv_performance = {
        'Other' : get_metrics_per_target(0),
        'GB' : get_metrics_per_target(1),
        'GBT' : get_metrics_per_target(2)
    }

    return cv_performance


def plot_performace(model, test_loader, model_args, load_from_dir=False, title='Test Dataset'):
    labels, preds = test(model, test_loader, model_args, load_from_dir)
    cm  = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)

    print(classification_report(labels, preds))

    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=['Other', 'GB', 'GBT'], yticklabels=['Other', 'GB', 'GBT'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{title} - Confusion Matrix - {len(preds)} Samples')
    plt.show(block=False)

    return report


def plot_model_accuracy_and_loss(train_loss_tot, test_loss_tot, train_acc_tot, test_acc_tot):
    
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 8))
    plt.subplot(211)
    plt.plot(train_loss_tot, marker='o', linestyle='--', color='b')
    plt.plot(test_loss_tot, marker='o', linestyle='--', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])
    plt.title('Loss')

    plt.subplot(212)
    plt.plot(train_acc_tot, marker='o', linestyle='--', color='b')
    plt.plot(test_acc_tot, marker='o', linestyle='--', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.title('Accuracy')
    plt.tight_layout()


