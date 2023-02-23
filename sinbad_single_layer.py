

from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import argparse
import ResNet as resnet
from utils import kNN_shrunk
from set_features import CumulativeSetFeatures
import wandb
from torchvision import transforms

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--version_name', default='example_run', type=str) #run name
parser.add_argument('--mvtype', default='short_breakfast_box', type=str) #mvtec class
parser.add_argument('--n_score', default=1, type=int) #number of nearest neighbours

parser.add_argument('--net', default='wide_resnet50_2') #net features to be used
parser.add_argument('--is_cpu', action='store_true', help='is_cpu') #is_cpu for kNN
parser.add_argument('--initial_res', default=1024, type=int)
parser.add_argument('--crop_size_ratio', default=0.999, type=float)
parser.add_argument('--crop_num_edge', default=4, type=int) #1 over the stride size (4 corresponds to a stride of 1/4)
parser.add_argument('--pyramid_level', default=14, type=float)

parser.add_argument('--n_projections', default=1000, type=int)
parser.add_argument('--n_quantiles', default=5, type=int)

parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--batch_size', default=32, type=float)
parser.add_argument('--shrinkage_factor', default=0.1, type=float) #shrinkage factor for whitening

args = parser.parse_args()

print("args",args)
fold_name = "dataset_loco"


mvtype = args.mvtype
mvtype_list = np.array(['breakfast_box_loco','juice_bottle_loco','pushpins_loco','screw_bag_loco','splicing_connectors_loco',
                                    'breakfast_box_struct','juice_bottle_struct','pushpins_struct','screw_bag_struct','splicing_connectors_struct'])
mv_num = np.where(mvtype_list == args.mvtype)[0]


stride_size = int(np.floor(args.initial_res / args.crop_num_edge))
crop_size = int(np.floor(args.initial_res*args.crop_size_ratio)) #only for naming convention
in_im_size = args.initial_res
img_size = in_im_size*in_im_size

batch_size = args.batch_size

wandb.init(project="mvtec_loco_sinbad_wandb_" + args.version_name, entity="nivcohen", config = args)
wandb.config = args


anom_maps_output_path = "../sinbad_runs/%s_pyramid_lvl_%d/%s"%\
                        (args.version_name, args.pyramid_level, mvtype)
os.makedirs(anom_maps_output_path,exist_ok=True)


if args.pyramid_level == 224:
    args.crop_size_ratio = 0.999
    crop_size = int(np.floor(args.initial_res*args.crop_size_ratio))
    n_channels = 3
if args.pyramid_level == 7:
    n_channels = 2048
if args.pyramid_level == 14:
    n_channels = 1024


if args.net == "wide_resnet50_2":
    net = resnet.wide_resnet50_2(pretrained=True, resnet_block = 1)

net.to('cuda')
net.eval()

transform_color = transforms.Compose([transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

args.epochs = 1



train_images = np.load('../%s/data_matrices/%s/train_data.npy'%(fold_name, args.mvtype))
valid_images = np.load('../%s/data_matrices/%s/valid_data.npy'%(fold_name, args.mvtype))
test_images = np.load('../%s/data_matrices/%s/test_data.npy'%(fold_name, args.mvtype))
image_level_label = np.load("../%s/data_matrices/%s/image_level_label.npy"%(fold_name, args.mvtype))

def get_super_patch_embeddings(im_from_load):
    with torch.no_grad():
        super_patches = im_from_load.unfold(2, crop_size, stride_size).unfold(3, crop_size, stride_size)
        super_patches = torch.reshape(super_patches,(3,super_patches.shape[2]*super_patches.shape[3],super_patches.shape[4],super_patches.shape[5]))
        super_patches = torch.transpose(super_patches, 0, 1)
        super_patches = F.interpolate(super_patches, (224, 224))

        if args.pyramid_level > 56:
            trans_clr = transforms.Compose([transforms.Resize(int(args.pyramid_level))])
            super_patches = trans_clr(super_patches)
            X = torch.reshape(super_patches, (super_patches.shape[0], super_patches.shape[1],-1))
            token = 0
        else:
            x, x77, x1414, x2828, x5656 = net(super_patches.cuda())

            if args.pyramid_level == 7:
                X = torch.reshape(x77, (x77.shape[0], x77.shape[1], x77.shape[2]*x77.shape[3]))
            if args.pyramid_level == 14:
                X = torch.reshape(x1414, (x1414.shape[0], x1414.shape[1], x1414.shape[2]*x1414.shape[3]))
            if args.pyramid_level == 28:
                X = torch.reshape(x2828, (x2828.shape[0], x2828.shape[1], x2828.shape[2]*x2828.shape[3]))
            if args.pyramid_level == 56:
                X = torch.reshape(x5656, (x5656.shape[0], x5656.shape[1], x5656.shape[2]*x5656.shape[3]))

            token = x

        return X, token

test_images = np.expand_dims(test_images, axis = 1)
list_testloader = torch.from_numpy(test_images)

valid_images = np.expand_dims(valid_images, axis = 1)
list_validloader = torch.from_numpy(valid_images)

train_images = np.expand_dims(train_images, axis = 1)
list_trainloader = torch.from_numpy(train_images)

L_train = len(list_trainloader)
L_valid = len(list_validloader)
L_test = len(list_testloader)


### sample to get dimensions
im_from_load = list_trainloader[0][:,:,:,:].cuda()
patch_descriptors, cls_tokens = get_super_patch_embeddings(im_from_load)

if args.pyramid_level == 224:
    anom_maps_output_file = os.path.join(anom_maps_output_path, "%.2f_lvl_anom_maps"%crop_size)
else:
    anom_maps_output_file = os.path.join(anom_maps_output_path, "%d_crop_anom_maps"%patch_descriptors.shape[0])

lbl_output_file = os.path.join(anom_maps_output_path, "lbl.py")


anom_map_valid = np.zeros((patch_descriptors.shape[0], L_valid))
anom_map = np.zeros((patch_descriptors.shape[0], L_test))

def get_mini_patches(L, list_loader, local_mini_patches_emb, i, transform = transform_color, epochs = 1):

    for epoch in range(epochs):
        for k_ind in range(L):
            with torch.no_grad():
                im_from_load = list_loader[k_ind][:,:,:,:].cuda()
                im_from_load = transform(im_from_load[0]).unsqueeze(0)
                patch_descriptors, cls_tokens = get_super_patch_embeddings(im_from_load)
                local_mini_patches_emb[epoch*len(list_loader) + k_ind] = patch_descriptors[i]


    return local_mini_patches_emb


def extract_radon_feaures(radon_extractor, local_mini_patches_emb, is_projection = True):
    ind_c = 0
    if is_projection:
        radon_feaures = np.zeros((len(local_mini_patches_emb), args.n_projections*args.n_quantiles))
    else:
        radon_feaures = np.zeros((len(local_mini_patches_emb), n_channels*args.n_quantiles))

    for batch_idx in range(int(np.ceil(len(local_mini_patches_emb)/batch_size))):
        batch_local_mini_patches_emb = local_mini_patches_emb[ind_c:int(ind_c+batch_size)]

        if ind_c == 0:
            radon_extractor.fit(batch_local_mini_patches_emb)
        batch_train_radon, _ = radon_extractor.forward(batch_local_mini_patches_emb)
        radon_feaures[ind_c:int(ind_c+batch_size)] = batch_train_radon

        ind_c = int(ind_c + batch_size)

    return radon_feaures


for i in range(patch_descriptors.shape[0]):
    print("crop num",i)
    local_train_mini_patches_emb = torch.zeros((L_train*args.epochs, patch_descriptors.shape[1], patch_descriptors.shape[2]))
    local_valid_mini_patches_emb = torch.zeros((L_valid, patch_descriptors.shape[1], patch_descriptors.shape[2]))
    local_test_mini_patches_emb = torch.zeros((L_test, patch_descriptors.shape[1], patch_descriptors.shape[2]))

    local_train_mini_patches_emb = get_mini_patches(L_train, list_trainloader, local_train_mini_patches_emb, i, transform = transform_color, epochs = args.epochs)
    local_valid_mini_patches_emb = get_mini_patches(L_valid, list_validloader, local_valid_mini_patches_emb, i)
    local_test_mini_patches_emb = get_mini_patches(L_test, list_testloader, local_test_mini_patches_emb, i)

    radon_extractor = CumulativeSetFeatures(local_train_mini_patches_emb.shape[1], n_projections=args.n_projections, n_quantiles=args.n_quantiles, is_projection=True)

    train_radon = extract_radon_feaures(radon_extractor, local_train_mini_patches_emb, is_projection=True)
    print("train_radon",train_radon.shape)

    valid_radon = extract_radon_feaures(radon_extractor, local_valid_mini_patches_emb, is_projection=True)
    print("valid_radon",valid_radon.shape)

    test_radon = extract_radon_feaures(radon_extractor, local_test_mini_patches_emb, is_projection=True)
    print("test_radon",test_radon.shape)

    print("scoring")

    is_whitening = True
    kNN_index = kNN_shrunk(train_radon, args.n_score, args.is_cpu,  is_whitening, is_vector = True, shrinkage_factor = args.shrinkage_factor)
    pred_score = kNN_index.score(test_radon)
    pred_score_valid = kNN_index.score(valid_radon)
    pred_score = pred_score[:,0]
    pred_score_valid = pred_score_valid[:,0]

    anom_map[i] = pred_score
    anom_map_valid[i] = pred_score_valid

np.savez(anom_maps_output_file, anom_map_test = anom_map, anom_map_valid = anom_map_valid)
np.save(lbl_output_file, image_level_label)



image_level_pred_max = np.max(anom_map, axis = 0)
image_level_pred_avg = np.mean(anom_map, axis = 0)

auc_avg = roc_auc_score(image_level_label, image_level_pred_avg)

print("ROC-AUC %.3f"%auc_avg)

wandb.log({
            "Image ROC-AUC avg": auc_avg,
})