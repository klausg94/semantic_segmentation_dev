import os
import logging
import datetime



def get_data_paths(train_folder, val_folder, classes_file):
    train_img_path = os.path.join(train_folder, "images")
    train_mask_path = os.path.join(train_folder, "labels")
    val_img_path = os.path.join(val_folder, "images")
    val_mask_path = os.path.join(val_folder, "labels")
    with open(classes_file, 'r') as f:
        classes = f.read().splitlines()
    classes_dict = {c:i for i,c in enumerate(classes)}
    return train_img_path, train_mask_path, val_img_path, val_mask_path, classes_dict, classes


def set_up_experiment(config):
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_folder = os.path.join(config['save_folder'], current_time + '_' + config['model_version'])
    assert os.path.isdir(config['save_folder'])
    if not os.path.isdir(experiment_folder):
        os.makedirs(experiment_folder)
    logger = get_logger_to_file(experiment_folder)
    return experiment_folder, logger
    
    
def get_logger_to_file(run_folder, logname = 'logs'):
    assert os.path.isdir(run_folder)
    logger = logging.getLogger('main')
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.DEBUG)

    log_filepath = os.path.join(run_folder, str(logname)+'.log')
    file_hdlr = logging.FileHandler(log_filepath)
    file_hdlr.setFormatter(formatter)
    file_hdlr.setLevel(logging.INFO)
    logger.addHandler(file_hdlr)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger