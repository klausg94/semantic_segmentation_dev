import os
import argparse

import yaml
import onnx
import torch

from model.model import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')

    args = parser.parse_args()
    config_file = args.config
    #config_file = "configs/config1.yml"
    with open(config_file) as f:
        config = yaml.load(f, yaml.Loader) 
    
    # MODEL
    num_classes = config["num_classes"]
    model_version = config['model_version']
    model = get_model(backbone=model_version, num_classes=num_classes).cpu()
    
    # LOAD MODEL WEIGHTS
    model_weight_path = args.weights
    state_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)

    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size)

    # ONNX EXPORT
    export_file = model_weight_path[:model_weight_path.rfind(".")] + ".onnx"
    dynamic_axes = None
    output_names = ['output']
    torch.onnx.export(model, img, export_file, verbose=False, opset_version=12, input_names=['images'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

    # Checks
    onnx_model = onnx.load(export_file)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print("ONNX export complete")

