import os
import yaml
import argparse
from collections import OrderedDict

import torch

from data_handling.data import get_data_loaders
from model.model import get_model, ModelEMA
from utils.experiment_setup import get_data_paths, set_up_experiment
from training_utils.optim import get_optimizer
from training_utils.losses import get_loss_function
from training_utils.evaluation import evaluate_model


def train(config):
    
    # DATA PATHS
    (train_img_path, 
     train_mask_path, 
     val_img_path, 
     val_mask_path, 
     classes_dict, 
     classes) = get_data_paths(config['train_folder'], config['val_folder'], config['classes_file'])
    
    config["train_img_path"] = train_img_path
    config["train_mask_path"] = train_mask_path
    config["val_img_path"] = val_img_path
    config["val_mask_path"] = val_mask_path
    config["classes_dict"] = classes_dict
    config["classes"] = classes
    
    # CREATE EXPERIMENT FOLDER AND LOGGER
    experiment_folder, logger = set_up_experiment(config)
    model_save_path = os.path.join(experiment_folder, config['model_version'] + config['save_add_string'] + '.pth')
    with open(os.path.join(experiment_folder, 'config.yml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    # MODEL
    num_classes = len(classes_dict.keys())
    device = torch.device('cuda')
    model_version = config['model_version']
    model = get_model(backbone=model_version, num_classes=num_classes)
    model.to(device)

    # model ema
    if "ema" in config.keys() and config["ema"]:
        ema = ModelEMA(model)
    else:
        ema = None
    
    # OPTIONALLY LOAD EXISTING WEIGHTS
    if config['load_prev']:
        model_weight_path = config['weight_load_path']
        state_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)

    # DATA LOADERS
    loader_train, loader_val = get_data_loaders(config)
    
    # OPTIMIZER
    optimizer = get_optimizer(model, config)
    
    # LOSS
    criterion = get_loss_function(config, num_classes, device)

    # TRAINING
    epochs=config["epochs"]
    eval_interval = config["eval_interval"]
    if eval_interval == -1:
        eval_interval = len(loader_train)
    print("EVAL INTERVAL ", eval_interval)

    save_str = model_save_path + '\n\n'
    if config is not None:
        save_str += str(config) + '\n\n'
    
    # INITIAL EVALUATION
    best_res_log = {}
    if loader_val is not None:
        eval_results = evaluate_model(model, criterion, loader_val, ignore_index=config['ignore_index'])
        eval_results = eval_results["eval_results"]
    else:
        eval_results = {'Loss':-1}
    for k,v in eval_results.items():
        best_res_log[k] = 0.
    res_str = "Epoch: {}/{}...".format(0, epochs) + ' '.join([k + ' : {:.4f}'.format(v) for k,v in eval_results.items()])
    save_str += res_str + '\n'
    #
    metric = 'acc'
    if metric in eval_results.keys():
        best_res_log[metric] = 0.
    metric = 'loss'
    if metric in eval_results.keys():
        best_res_log[metric] = 999.
    metric = 'bacc'
    if metric in eval_results.keys():
        best_res_log[metric] = 0.
    metric = 'iou'
    if metric in eval_results.keys():
        best_res_log[metric] = 0.
    metric = 'tpr'
    if metric in eval_results.keys():
        best_res_log[metric] = 0.
    metric = 'prec'
    if metric in eval_results.keys():
        best_res_log[metric] = 0.
    metric = 'f1'
    if metric in eval_results.keys():
        best_res_log[metric] = 0.
    #
    logger.info(res_str)
    
    steps = 0
    device = next(iter(model.parameters())).device
    early_stopping = config["early_stopping"]
    early_stopping_counter = 0
    best_score_saved = 0.
    for e in range(epochs):
        model.train()
        if not config['train_backbone']:
            model.model.encoder.eval()
        
        if e in config["lr_red_epochs"]:
            lr_red_factor = config["lr_red_factors"][config["lr_red_epochs"].index(e)]
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_red_factor
                
        for inputs, labels in loader_train:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(inputs)
            if isinstance(output, OrderedDict):
                output = output['out']
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            steps+=1
            if ema:
                ema.update(model)
                model_to_eval = ema.ema
            else:
                model_to_eval = model
            
            if steps % eval_interval == 0:
                if loader_val is not None:
                    eval_results = evaluate_model(model_to_eval, criterion, loader_val, ignore_index=config['ignore_index'])
                    eval_results = eval_results["eval_results"]
                else:
                    eval_results = {'Loss':-1}
                metric = 'acc'
                if metric in eval_results.keys() and eval_results[metric] > best_res_log[metric]:
                    best_res_log[metric] = eval_results[metric]
                    torch.save(model_to_eval.state_dict(), model_save_path[:-4]+'_' + metric + '.pth')
                metric = 'loss'
                if metric in eval_results.keys() and eval_results[metric] < best_res_log[metric]:
                    best_res_log[metric] = eval_results[metric]
                    torch.save(model_to_eval.state_dict(), model_save_path[:-4]+'_' + metric + '.pth')
                metric = 'bacc'
                if metric in eval_results.keys() and eval_results[metric] > best_res_log[metric]:
                    best_res_log[metric] = eval_results[metric]
                    torch.save(model_to_eval.state_dict(), model_save_path[:-4]+'_' + metric + '.pth')
                metric = 'iou'
                if metric in eval_results.keys() and eval_results[metric] > best_res_log[metric]:
                    best_res_log[metric] = eval_results[metric]
                    torch.save(model_to_eval.state_dict(), model_save_path[:-4]+'_' + metric + '.pth')
                metric = 'tpr'
                if metric in eval_results.keys() and eval_results[metric] > best_res_log[metric]:
                    best_res_log[metric] = eval_results[metric]
                    torch.save(model_to_eval.state_dict(), model_save_path[:-4]+'_' + metric + '.pth')
                metric = 'prec'
                if metric in eval_results.keys() and eval_results[metric] > best_res_log[metric]:
                    best_res_log[metric] = eval_results[metric]
                    torch.save(model_to_eval.state_dict(), model_save_path[:-4]+'_' + metric + '.pth')
                metric = 'f1'
                if metric in eval_results.keys() and eval_results[metric] > best_res_log[metric]:
                    best_res_log[metric] = eval_results[metric]
                    torch.save(model_to_eval.state_dict(), model_save_path[:-4]+'_' + metric + '.pth')
                
                
                res_str = "Epoch: {}/{}...".format(e+1, epochs) + ' '.join([k + ' : {:.4f}'.format(v) for k,v in eval_results.items()])
                logger.info(res_str)
                save_str += res_str + '\n'
                
                # EARLY STOPPING
                if early_stopping:
                    early_stopping_metric = config["early_stopping_metric"]
                    if early_stopping_metric in eval_results.keys() and best_res_log[early_stopping_metric] > best_score_saved and e>5:
                        best_score_saved = best_res_log[early_stopping_metric]
                        early_stopping_counter = 0
                    elif e>5:
                        early_stopping_counter += 1
                    if early_stopping_counter >= config["early_stopping_counts"]:
                        logger.info("EARLY STOPPING")
                        break
        if early_stopping:                
            if early_stopping_counter >= config["early_stopping_counts"]:
                logger.info("EARLY STOPPING")
                break
    
    logger.info("\n\nBEST SCORES  " + ' '.join([k + ' : {:.4f}'.format(v) for k,v in best_res_log.items()]))


if __name__ == '__main__':
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str)
    #args = parser.parse_args()
    #config_file = args.config
    config_file = "configs/config1.yml"
    with open(config_file) as f:
        config = yaml.load(f, yaml.Loader) 

    train(config)
    