import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
import argparse
from scipy.stats import trim_mean

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--mvtype', default='breakfast_box_loco', type=str) #mvtec class / loco / struct / all
parser.add_argument('--version', default='/path/to/sinbad_runs/results/ver1_pyramid_lvl_#', type=str)
parser.add_argument('--version_224', default='/path/to/sinbad_runs/results/ver1_pyramid_lvl_#', type=str)
parser.add_argument('--rep_num_224', default=100, type=int)
parser.add_argument('--lambda_factor', default=0.1, type=float)

args = parser.parse_args()

fold_path = "/cs/labs/peleg/nivc/Palach/"

if args.mvtype == "all":
    mvtype_list = np.array(['breakfast_box_loco','juice_bottle_loco','pushpins_loco','screw_bag_loco','splicing_connectors_loco',
                                'breakfast_box_struct','juice_bottle_struct','pushpins_struct','screw_bag_struct','splicing_connectors_struct'])
elif args.mvtype == "loco":
    mvtype_list = np.array(['breakfast_box_loco','juice_bottle_loco','pushpins_loco','screw_bag_loco','splicing_connectors_loco'])
elif args.mvtype == "struct":
    mvtype_list = np.array(['breakfast_box_struct','juice_bottle_struct','pushpins_struct','screw_bag_struct','splicing_connectors_struct'])
else:
    mvtype_list = [args.mvtype]

reg_eps = 0.00001


roc_mean_list = []
roc_max_list = []
roc_loc_list = []
roc_clr_list = []
roc_mean_w_clr_list = []
roc_max_w_clr_list = []

mvtypes_roc = []



def get_anom_map_224(resized_anom_maps_res_lvls, set_string = "anom_map_test"):

    anom_maps = []
    lvl_version = args.version_224.replace("#", "%d"%(224))
    res_fold_temp = os.path.join(fold_path, lvl_version + "/rep_num_#", mvtype)

    for j in range(0,args.rep_num_224): #range(args.num_of_224_reps):
        res_fold = res_fold_temp.replace("#", "%d"%(j))
        for file_name in (os.listdir(res_fold)):
            file_path = os.path.join(res_fold, file_name)

            if ("anom_maps" in file_name):

                array = np.load(file_path)
                array = array[set_string]
                anom_maps.append(array)

    anom_maps = np.array(anom_maps)
    return anom_maps[:,0]

def get_anom_map(resized_anom_maps_res_lvls, set_string = "anom_map_test", rep_num = 0):

    anom_maps_sized_resized_agg_res_lvls = []

    res_lvls = ([7,14])

    for res_lvl in res_lvls:
        lvl_version = args.version.replace("#", "%d"%(res_lvl))
        res_fold = os.path.join(fold_path, lvl_version + "/rep_num_%d"%(rep_num), mvtype)

        anom_maps = []
        anom_lens = []

        res_folds = os.listdir(res_fold)
        res_folds.sort()

        for j, file_name in enumerate(res_folds):
            file_path = os.path.join(res_fold, file_name)

            if ("anom_maps" in file_name):

                            i = int(file_name.split("_")[0])

                            array = np.load(file_path)
                            array = array[set_string]

                            anom_maps.append(array)
                            anom_lens.append(i)

            else:
                    gt_labels = np.load(file_path)

        max_length = int(np.sqrt(np.max(anom_lens)))
        resized_anom_maps = np.zeros((anom_maps[0].shape[1],  len(anom_maps), max_length, max_length)) # samplex X super patch size X len X len

        anom_maps_sized_resized_agg_super_pixels = []

        for i, length in enumerate(anom_lens):

            anom_maps_sized = np.transpose(anom_maps[i], (1,0))

            length_i = int(np.sqrt(np.max(length)))

            for img_ind in range(anom_maps[0].shape[1]):
                squared_map = np.reshape(anom_maps_sized[img_ind],(length_i,length_i))
                squared_map_resized = cv2.resize(squared_map, dsize=(max_length, max_length), interpolation=cv2.INTER_CUBIC)
                resized_anom_maps[img_ind, i] = squared_map_resized

        resized_anom_maps_res_lvls.append(np.stack(resized_anom_maps))
        anom_maps_sized_resized_agg_res_lvls.append(anom_maps_sized_resized_agg_super_pixels)

    resized_anom_maps_res_lvls = np.stack(resized_anom_maps_res_lvls) # res_lvls X samplex X super patch size X len X len
    resized_anom_maps_res_lvls = np.transpose(resized_anom_maps_res_lvls, (1, 0, 2, 3, 4)) # samplex X res_lvls X super patch size X len X len

    resized_anom_maps_res_lvls = np.mean(resized_anom_maps_res_lvls, 3)
    resized_anom_maps_res_lvls = np.mean(resized_anom_maps_res_lvls, 3)

    return resized_anom_maps_res_lvls, gt_labels

for mvtype in mvtype_list:
    resized_anom_maps_res_lvls = []
    resized_anom_maps_res_lvls_valid = []

    resized_anom_maps_res_lvls, gt_labels = get_anom_map(resized_anom_maps_res_lvls, set_string = "anom_map_test")
    resized_anom_maps_res_lvls_valid, _ = get_anom_map(resized_anom_maps_res_lvls_valid, set_string = "anom_map_valid")

    resized_anom_maps_res_lvls_224 = get_anom_map_224(resized_anom_maps_res_lvls, set_string = "anom_map_test")
    resized_anom_maps_res_lvls_valid_224 = get_anom_map_224(resized_anom_maps_res_lvls_valid, set_string = "anom_map_valid")

    for i in range(resized_anom_maps_res_lvls.shape[1]):
        for j in range(resized_anom_maps_res_lvls.shape[2]):

            norm_fac = np.mean(resized_anom_maps_res_lvls_valid[:,i,j]) + 0.00001
            resized_anom_maps_res_lvls[:,i,j] = resized_anom_maps_res_lvls[:,i,j]/norm_fac + 0.00001

            roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls[:,i,j])

    for i in range(resized_anom_maps_res_lvls_224.shape[0]):

        norm_fac = np.mean(resized_anom_maps_res_lvls_valid_224[i]) + reg_eps
        resized_anom_maps_res_lvls_224[i] = resized_anom_maps_res_lvls_224[i]/norm_fac + reg_eps
        roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls_224[i])

    resized_anom_maps_res_lvls_224_mean = np.median(resized_anom_maps_res_lvls_224, 0)

    roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls_224_mean)

    resized_anom_maps_res_lvls_pre_mean = resized_anom_maps_res_lvls
    resized_anom_maps_res_lvls_mean = np.mean(resized_anom_maps_res_lvls, axis = 2)
    resized_anom_maps_res_lvls_mean = np.mean(resized_anom_maps_res_lvls_mean, axis = 1)

    roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls_mean + resized_anom_maps_res_lvls_224_mean*args.lambda_factor)
    mvtypes_roc.append(roc)

for i, roc in enumerate(mvtypes_roc):
    print("Accuracy on %s is %.2f" % (mvtype_list[i],roc))

print("mean_roc %.3f"%np.mean(mvtypes_roc))

