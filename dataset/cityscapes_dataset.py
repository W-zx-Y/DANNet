import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from dataset.transforms import *
import torchvision.transforms as standard_transforms


class cityscapesDataSet(data.Dataset):
    def __init__(self, args, root, list_path, max_iters=None, set='val'):
        self.root = root
        self.list_path = list_path

        train_input_transform = []
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        cityscape_transform_list = [joint_transforms.RandomSizeAndCrop(args.input_size, False, pre_size=None,
                                                                       scale_min=0.5, scale_max=1.0, ignore_index=255),
                                    joint_transforms.Resize(args.input_size),
                                    joint_transforms.RandomHorizontallyFlip()]
        self.joint_transform = joint_transforms.Compose(cityscape_transform_list)

        self.target_transform = extended_transforms.MaskToTensor()
        self.transform = standard_transforms.Compose(train_input_transform)

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name)).replace("leftImg8bit", "gtFine_labelIds")
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        label = np.asarray(label, np.uint8)

        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        label = Image.fromarray(label_copy.astype(np.uint8))

        if self.joint_transform is not None:
            image, label = self.joint_transform(image, label)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        size = image.shape
        return image, label, np.array(size), name


