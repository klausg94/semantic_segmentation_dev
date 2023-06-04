# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:35:07 2021

@author: Z006AIKC
"""
import os
import glob
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

from PIL import Image
import numpy as np

from data import SegmentationDataset
from utils import get_model, train_model, evaluate_model, plot_confusion_matrix, get_data_paths, Augmented_Predictor
import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':
    model_paths = glob.glob("D:\\Projects\\CrackDetection\\training_output\\*\\*\\*_f1.pth")[-1:]
    
    for model_weight_path in model_paths:
        #model_weight_path = "D:\\Projects\\CrackDetection\\training_output\\20220331_143816_deeplabv3_mobilenet_v3_large\\deeplabv3_mobilenet_v3_large__bacc.pth"
        eval_folder = os.path.join(os.path.split(model_weight_path)[0], 'evaluation')
        if not os.path.isdir(eval_folder):
            os.makedirs(eval_folder)
        config_file = os.path.join(os.path.split(model_weight_path)[0], 'config.yml')
        with open(config_file) as f:
            config = yaml.load(f, yaml.Loader)
            
        # PATHS
        data_folder = config['data_folder']
        train_img_path, train_mask_path, val_img_path, val_mask_path, classes_dict, classes = get_data_paths(data_folder)
        
        # MODEL
        model_version = config['model_version']
        model = get_model(backbone=model_version, num_classes=len(classes_dict.keys()))
        
        # LOAD WEIGHTS
        #######
        state_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))
        if list(state_dict.keys())[0].startswith("model"):
            model.load_state_dict(state_dict, strict=True)
        else:
            model.model.load_state_dict(state_dict, strict=True)
        torch.save(model.model.state_dict(), os.path.join(os.path.split(model_weight_path)[0], "MODEL.pth"))
        
        #######
        # DATA AUGMENTED PREDICTOR
        model = Augmented_Predictor(model).cuda()
        model.eval()
        
        # DATA
        batch_size = 1
        resolution_div = config['resolution_div']
        
        train_dataset = SegmentationDataset(train_img_path, train_mask_path, 
                                            split='val', resolution_div=resolution_div, 
                                            norm_means=config['norm_means'], norm_stds=config['norm_stds'], 
                                            reshaping_shape=config['reshaping_shape'])
        loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        
        val_dataset = SegmentationDataset(val_img_path, val_mask_path, 
                                          split='val', resolution_div=resolution_div, 
                                          norm_means=config['norm_means'], norm_stds=config['norm_stds'], 
                                          reshaping_shape=config['reshaping_shape'])
        loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        
        # EVALUATION
        criterion = nn.CrossEntropyLoss(ignore_index=config['ignore_index'])
        if config['binary_sigmoid']:
            criterion = nn.BCELoss()
        eval_results, conf_m, unique_classes = evaluate_model(model, criterion, loader_val, downscaling=0.5, ignore_index=config['ignore_index'],
                                                              return_confusion_matrix=True)
        res_str = ' '.join([k + ' : {:.4f}'.format(v) for k,v in eval_results.items()])
        with open(os.path.join(eval_folder, 'metrics.txt'), 'w') as f:
            f.write(res_str)
        print(model_weight_path)
        print(res_str)
        class_names = [classes[c] for c in unique_classes]
        plot_confusion_matrix(conf_m, class_names, title='Confusion matrix normalized', save_folder=eval_folder)