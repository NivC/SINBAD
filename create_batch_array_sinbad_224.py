import os
import numpy as np

out_path = "/cs/labs/peleg/nivc/Palach/iclr2024/sinbad_runs/sbatches_224/"

dataset_names = ['breakfast_box_loco','juice_bottle_loco','pushpins_loco','screw_bag_loco','splicing_connectors_loco',
                 'breakfast_box_struct','juice_bottle_struct','pushpins_struct','screw_bag_struct','splicing_connectors_struct']


n_projections_list = [10]
n_quantiles_list = [5]


pyramid_level_list = [224]
patch_size_ratio_list = [0.99]
shrinkage_factor_list = [1]
super_patch_n_list = [4]
rep_list = np.arange(0,100)
name_list = []

for dataset_name in dataset_names:
    for n_projections in n_projections_list:
        for n_quantiles in n_quantiles_list:
                for pyramid_level in pyramid_level_list:
                    for patch_size_ratio in patch_size_ratio_list:
                            for shrinkage_factor in shrinkage_factor_list:
                                    for super_patch_n in super_patch_n_list:
                                        for rep in rep_list:
                                            name_list.append([dataset_name, n_projections, n_quantiles, pyramid_level, patch_size_ratio, shrinkage_factor,super_patch_n, rep])

for i in range(len(name_list)):

    f  = open(out_path + str(i) + '.sh', 'w')
    f.write("""#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib/llvm-7/lib/
cd /cs/labs/peleg/nivc/Palach/iclr2024/SINBAD
source /cs/labs/peleg/nivc/Palach/pre_git/venv/bin/activate.csh
python sinbad_single_layer.py --version_name "ver1"  --mvtype %s --n_projections %d --n_quantiles %d   --net wide_resnet50_2   --pyramid_level %d  --crop_size_ratio %f --shrinkage_factor %f   --crop_num_edge  %d --rep %d """
            %( name_list[i][0], name_list[i][1], name_list[i][2], name_list[i][3], name_list[i][4], name_list[i][5], name_list[i][6], name_list[i][7]))
    f.close()


f  = open(out_path + "batch_master" + '.sh', 'w')
f.write("sbatch --array=0-%d%s20 --exclude=cyril-01,ampere-01,binky-01,binky-02,binky-03,binky-04,binky-05,binky-06,arion-01,arion-02,drape-01,drape-02 --gres=gpu:1,vmem:10g --mem=18000m -c2 --time=1-12 -A yedid /cs/labs/peleg/nivc/Palach/iclr2024/sinbad_runs/sbatches_224/bishop.sh"%(len(name_list),'%'))



f  = open(out_path + "bishop" + '.sh', 'w')
f.write("#! /bin/bash\n")
f.write("sh %s/$SLURM_ARRAY_TASK_ID.sh\n" %(out_path))
f.close()
