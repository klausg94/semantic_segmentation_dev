import os
import glob

import torch
import cv2


__all__ = ["torch_masks_to_png", "crop_horizontal", "bgr2gray", "equalize_hist", "adaptive_hist_eq", "get_image_means_and_stds"]



def torch_masks_to_png(torch_files, output_folder):
    for torch_file in torch_files:
        img = torch.load(torch_file)
        img = img.numpy().astype('uint8')
        png_name = os.path.split(torch_file[:torch_file.rfind(".")] + ".png")[1]
        png_name = os.path.join(output_folder, png_name)
        cv2.imwrite(png_name, img)


def crop_horizontal(img_files, out_folder, cropping=(0,0)):
    for file in img_files:
        img = cv2.imread(file)
        img = img[:,cropping[0]:-cropping[1]]
        out_path = os.path.join(out_folder, os.path.split(file)[1])
        cv2.imwrite(out_path, img)


def bgr2gray(img_files, out_folder):
    for file in img_files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out_path = os.path.join(out_folder, os.path.split(file)[1])
        cv2.imwrite(out_path, img)


def equalize_hist(img_files, out_folder):
    for file in img_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.equalizeHist(img)
        out_path = os.path.join(out_folder, os.path.split(file)[1])
        cv2.imwrite(out_path, img)


def adaptive_hist_eq(img_files, out_folder):
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(20, 20))
    for file in img_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = clahe.apply(img)
        out_path = os.path.join(out_folder, os.path.split(file)[1])
        cv2.imwrite(out_path, img)


def get_image_means_and_stds(img_files):
    means = []
    stds = []
    for file in img_files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        means.append(img.mean(axis=(0,1)))
        stds.append(img.std(axis=(0,1)))
    means = sum(means) / len(means)
    stds = sum(stds) / len(stds)
    print("MEAN:", means, "STD:",stds)