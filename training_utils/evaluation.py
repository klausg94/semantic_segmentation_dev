import os
from collections import OrderedDict

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, jaccard_score
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils



def evaluate_model(net, criterion, val_loader, ignore_index=None, binary_eval=True, 
                   return_confusion_matrix=False, downscaling=0.5):
    net.eval()
    eval_losses = []
    equals = []
    totals = []
    preds = []
    trues = []
    IoUs = []
    device = next(iter(net.parameters())).device
    with torch.no_grad():
        for val_batch, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = net(inputs)
            if isinstance(output, OrderedDict):
                output = output['out']
            
            eval_loss = criterion(output, labels)
            eval_losses.append(eval_loss)
            
            predictions = output.argmax(dim=1)

            labels = labels.cpu().numpy()
            predictions = predictions.cpu().numpy()
                
            # Downscale for computational feasability
            if downscaling != 1.:
                p_list = []
                t_list = []
                for label, pred in zip(labels, predictions):
                    t_list.append(downscale_class_mask(label, downscaling))
                    p_list.append(downscale_class_mask(pred, downscaling))
                labels = np.stack(t_list, axis=0)
                predictions = np.stack(p_list, axis=0)
            
            labels_fl = labels.flatten()
            output_fl = predictions.flatten()
            
            # Ignore Index
            if ignore_index is not None:
                if isinstance(ignore_index, (int, float)):
                    ignore_index = [ignore_index]
                idx = ~np.isin(labels_fl, ignore_index)
                labels_fl = labels_fl[idx]
                output_fl = output_fl[idx]
            
            equals.append((labels_fl == output_fl).sum().item())
            totals.append(labels_fl.size)
            
            preds += output_fl.tolist()
            trues += labels_fl.tolist()
            IoUs.append(compute_iou(output_fl, labels_fl))
            
            #if val_batch == 30:
                #break
    eval_losses = sum(eval_losses) / len(eval_losses)
    eval_losses =  eval_losses.item()
    acc = sum(equals) / sum(totals)
    iou = sum(IoUs) / len(IoUs)
    
    bacc = balanced_accuracy_score(trues, preds)
    #bacc = 0.
    eval_results = {'loss':eval_losses, 'acc':acc, 'bacc':bacc, 'iou':iou}
    
    if binary_eval and output.shape[1]==2:
        cm = compute_confusion_matrix(trues, preds, normalize=None)
        tp = cm[1].sum()
        if tp > 0:
            tpr = 1. * cm[1,1] / tp
        else:
            tpr = 0.
        pr = cm[:,1].sum()
        if pr > 0:
            prec = 1. * cm[1,1] / pr
        else:
            prec = 0
        if (tpr + prec) > 0:
            f1score = 2 * (tpr * prec) / (tpr + prec)
        else:
            f1score = 0.
        
        eval_results['tpr'] = tpr
        eval_results['prec'] = prec
        eval_results['f1'] = f1score
    
    if return_confusion_matrix:
        cm = compute_confusion_matrix(trues, preds, normalize='true')
        unique_classes = set(np.unique(preds).tolist() + np.unique(trues).tolist())
        return {"eval_results": eval_results, "confusion_matrix":cm, "unique_classes":unique_classes}
    
    return {"eval_results": eval_results}


def downscale_class_mask(mask, factor):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    shape = mask.shape[-2:]
    shape = (int(shape[-2] * factor), int(shape[-1] * factor))
    return cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_NEAREST)


def compute_confusion_matrix(targets, predictions, normalize='true'):
    if isinstance(targets,torch.Tensor):
        targets = targets.numpy()
    if isinstance(predictions,torch.Tensor):
        predictions = predictions.numpy()
    conf_m = confusion_matrix(targets, predictions, normalize=normalize)
    return conf_m


def compute_iou(y_pred, y_true):
    return jaccard_score(y_pred.flatten(), y_true.flatten(), average='macro')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          save_folder=None):
    import itertools
    if cmap is None:
        cmap = plt.get_cmap('Greens')
    # REMOVE ZERO COLUMNS
    sums = cm.sum(axis=1)
    zero_idx = np.where(sums == 0)[0].tolist()
    for index in sorted(zero_idx, reverse=True):
        del target_names[index]
    cm = np.delete(cm, zero_idx, axis=0)
    cm = np.delete(cm, zero_idx, axis=1)
    #####################

    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
        
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]*1),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.ylabel('Target')
    plt.xlabel('Prediction'
               #+ '\naccuracy={:0.4f}'.format(accuracy)
               )
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, 'confusion_matrix'))
    #plt.show()
    

def remove_outlier_points(mask):
    mask_copy = mask.numpy().copy()
    idx0, idx1 = np.where(mask==1)
    mask_copy[idx0, idx1] = 0
    cnts = cv2.findContours(np.uint8(mask_copy), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    cleaned_mask = np.ones_like(mask_copy)
    cleaned_mask[y : y + h, x : x + w] = mask_copy[y : y + h, x : x + w]
    return torch.from_numpy(cleaned_mask)