import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import PIL.Image as Image
import os


def default_loader(path):
    return Image.open(path).convert('RGB')

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class MyDataset(Dataset):



    def __init__(self, parent_path, which_set, class_name, anom_type, img_size=1024, img_resize=256, is_loco = True):

        transform =  transforms.Compose([
                transforms.Resize((img_resize,img_resize)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ])

        if is_loco:
            data_path = '../dataset_loco/'
        else:
            data_path = '../dataset_mvtec/'

        if which_set == 'train':
            fold_path = os.path.join(parent_path, data_path, class_name, "train")
        if which_set == 'validation':
            fold_path = os.path.join(parent_path, data_path, class_name, "validation")
        if which_set == 'test':
            fold_path = os.path.join(parent_path, data_path, class_name, "test")

        dataset = torchvision.datasets.ImageFolder(fold_path, transform)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=0,drop_last=False)



        target_list = np.zeros(len(trainloader))

        imgs = torch.zeros((len(trainloader),3, img_size, img_size))
        label_list = torch.zeros((len(trainloader)))
        is_relevant_list = np.zeros((len(trainloader)))

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            if (which_set is 'test'):
                if ((os.path.join('/test/',anom_type) in dataset.imgs[batch_idx][0]) or ('/test/good' in dataset.imgs[batch_idx][0])) or (anom_type == 'all'):
                    is_relevant_list[batch_idx] = 1
                else:
                    is_relevant_list[batch_idx] = 0
            else:
                is_relevant_list[batch_idx] = 1



            if (which_set is 'test') and (int(dataset.imgs[batch_idx][1]) > 0):  #If there is gt mask: anomlous test data
                label = 1
            else:
                label = 0


            imgs[batch_idx] = inputs[0]
            target_list[batch_idx] = (targets.item())

            label_list[batch_idx] = label

        self.targets = np.array(target_list)
        self.imgs = imgs

        relevant_inds = np.where(is_relevant_list==1)
        self.targets = np.array(label_list)[relevant_inds]
        self.imgs = imgs[relevant_inds]

    def __getitem__(self, index):

        img = self.imgs[index]
        label = self.targets[index]

        return img, label

    def __len__(self):
        return len(self.imgs)


def get_mvt_loader(parent_path, which_set = 'train', class_name = "breakfast_box", anom_type = "logical_anomalies", img_size=1024, img_resize=1024, is_loco = True):

    mvt_data_in = MyDataset(parent_path, which_set,  class_name, anom_type, img_size, img_resize, is_loco = is_loco)

    mvt_loader = torch.utils.data.DataLoader(
        mvt_data_in,
        batch_size=1, shuffle=False,
        num_workers=0)

    return mvt_loader


