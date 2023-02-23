import os

import load_mvtec_loco as mvt
import argparse
import pathlib
import torch
import numpy as np
import torch.nn.functional as F

def resize_array(new_img_size, in_array):
    array_new = torch.zeros(in_array.shape)
    array_interp = F.interpolate(in_array, (int(new_img_size[0]), int(new_img_size[1])))
    array_new[:,:,:int(new_img_size[0]),:int(new_img_size[1])] = array_interp
    return array_new

parent_path = pathlib.Path(__file__).parent.absolute()

print("parent_path",parent_path)
parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--im_size_before_crop', default=1024, type=int)
parser.add_argument('--im_size', default=1024, type=int)
args = parser.parse_args()

data_matrices_path = "../dataset_loco/data_matrices"

for mvtype in (['breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors']):
    print(mvtype)

    if "pushpins" in mvtype:
        img_aspects = [1700,1000]
    if "screw_bag" in mvtype:
        img_aspects = [1600,1100]
    if "splicing_connectors" in mvtype:
        img_aspects = [1700,850]
    if "juice_bottle" in mvtype:
        img_aspects = [800,1600]
    if "breakfast_box" in mvtype:
        img_aspects = [1600,1280]
    if "breakfast_box_short" in mvtype:
        img_aspects = [1600,1280]
    if "pushpins_short" in mvtype:
        img_aspects = [1700,1000]
    if "splicing_connectors_short" in mvtype:
        img_aspects = [1700,850]

    img_aspects = [img_aspects[1], img_aspects[0]]
    aspect_large_side = np.max(img_aspects)
    size_ratio = aspect_large_side/args.im_size
    img_aspects = img_aspects/size_ratio

    for anom_type in ['logical_anomalies','structural_anomalies']:

        if anom_type == "logical_anomalies":
            name_suffix = "loco"
        if anom_type == "structural_anomalies":
            name_suffix = "struct"

        out_path = os.path.join(data_matrices_path, "%s_%s"%(mvtype, name_suffix))
        os.makedirs(out_path, exist_ok=True)

        trainloader = mvt.get_mvt_loader(parent_path, 'train', mvtype, anom_type, args.im_size, args.im_size_before_crop, is_loco = True)
        validloader = mvt.get_mvt_loader(parent_path, 'validation', mvtype, anom_type, args.im_size, args.im_size_before_crop, is_loco = True)
        testloader = mvt.get_mvt_loader(parent_path, 'test', mvtype, anom_type, args.im_size, args.im_size_before_crop, is_loco = True)

        list_trainloader = torch.zeros((len(trainloader),3, args.im_size, args.im_size))
        list_validloader = torch.zeros((len(validloader),3, args.im_size, args.im_size))
        list_testloader = torch.zeros((len(testloader),3, args.im_size, args.im_size))

        image_level_label = np.zeros(len(testloader))


        for batch_idx, (img, label) in enumerate(trainloader):
            list_trainloader[batch_idx] = (img)

        for batch_idx, (img, label) in enumerate(validloader):
            list_validloader[batch_idx] = (img)

        for batch_idx, (img, label) in enumerate(testloader):
            list_testloader[batch_idx] = (img)
            image_level_label[batch_idx] = label

        list_trainloader = resize_array(img_aspects, list_trainloader)
        list_validloader = resize_array(img_aspects, list_validloader)
        list_testloader = resize_array(img_aspects, list_testloader)


        print("list_trainloader",list_trainloader.shape)

        np.save(os.path.join(out_path, "train_data.npy"),list_trainloader.numpy())
        np.save(os.path.join(out_path, "valid_data.npy"),list_validloader.numpy())
        np.save(os.path.join(out_path, "test_data.npy"),list_testloader.numpy())
        np.save(os.path.join(out_path, "image_level_label.npy"),image_level_label)
