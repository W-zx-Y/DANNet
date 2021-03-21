import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from dataset.transforms import *
import torchvision.transforms as standard_transforms


class zurich_pair_DataSet(data.Dataset):
    def __init__(self, args, root, list_path, max_iters=None, set='val', joint_transform=None):
        self.root = root
        self.list_path = list_path

        zurich_transform_list = [joint_transforms2.RandomSizeAndCrop(args.input_size_target, False, pre_size=None,
                                                                     scale_min=0.9, scale_max=1.1, ignore_index=255),
                                 joint_transforms2.Resize(args.input_size),
                                 joint_transforms2.RandomHorizontallyFlip()]
        self.joint_transform = joint_transforms2.Compose(zurich_transform_list)

        train_input_transform = []
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        self.target_transform = extended_transforms.MaskToTensor()
        self.transform = standard_transforms.Compose(train_input_transform)

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for pair in self.img_ids:
            night, day = pair.split(",")
            img_night = osp.join(self.root, "%s" % (night)+"_rgb_anon.png")
            img_day = osp.join(self.root, "%s" % (day)+"_rgb_anon.png")
            self.files.append({
                "img_night": img_night,
                "img_day": img_day,
                "name": night+"_rgb_anon.png"
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_n = Image.open(datafiles["img_night"]).convert('RGB')
        image_d = Image.open(datafiles["img_day"]).convert('RGB')

        name = datafiles["name"]

        if self.joint_transform is not None:
            image_n, image_d = self.joint_transform(image_n, image_d)
        if self.transform is not None:
            image_n = self.transform(image_n)
            image_d = self.transform(image_d)

        size = image_n.shape
        return image_n, image_d, np.array(size), name

