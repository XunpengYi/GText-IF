import os
from pickletools import uint8
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_root = setting['rgb_root']
        self._x_root = setting['x_root']
        self._rgb_gt_root = setting['rgb_gt_root']
        self._x_gt_root = setting['x_gt_root']

        self._text_root = setting['text_root']

        self._label_root = setting['label_root']
        self._label_transform = setting['label_transform']

        self.class_names = setting['class_names']

        self._img_path_list = self._get_file_list(self._rgb_root)
        self.preprocess = preprocess
        self.text_input_mode = setting['text_input_mode']

    def __len__(self):
        return len(self._img_path_list)

    def __getitem__(self, index):
        tmp_path = self._img_path_list[index]
        name = tmp_path.split('/')[-1].split('.')[0]

        rgb_path = os.path.join(self._rgb_root, name + '.png')
        x_path = os.path.join(self._x_root, name + '.png')
        rgb_gt_path = os.path.join(self._rgb_gt_root, name + '.png')
        x_gt_path = os.path.join(self._x_gt_root, name + '.png')

        text_path = os.path.join(self._text_root, name + '.txt')
        label_path = os.path.join(self._label_root, name + '.png')

        assert os.path.exists(rgb_path), rgb_path + ' does not exist'
        assert os.path.exists(x_path), x_path + ' does not exist'
        assert os.path.exists(rgb_gt_path), rgb_gt_path + ' does not exist'
        assert os.path.exists(x_gt_path), x_gt_path + ' does not exist'

        assert os.path.exists(text_path), text_path + ' does not exist'
        assert os.path.exists(label_path), label_path + ' does not exist'

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)
        x = self._open_image(x_path, cv2.COLOR_BGR2RGB)
        rgb_gt = self._open_image(rgb_gt_path, cv2.COLOR_BGR2RGB)
        x_gt = self._open_image(x_gt_path, cv2.COLOR_BGR2RGB)
        
        if self.text_input_mode == "saved":
            # Note: When using saved text inputs, image cropping and resizing may cause
            # unexpected text-image mismatches. Such misalignment can confuse the model
            # and potentially degrade performance. 
            text = str(self._open_text(text_path))
        else:
            # Note: The online text data generation module is not implemented by default in this solution. 
            # Since different API providers may vary in authentication mechanisms, request protocols, and response formats, 
            # please refer to the official documentation of the selected API service to implement the text generation and data loading workflow accordingly. 
            # This solution does not provide any specific API usage examples.

            # Modified by your own API text generation.

            raise NotImplementedError(
                "Online text generation is not implemented by default in this solution. "
                "Please implement your own text generation module based on the API service you intend to use. "
                "Different API providers may have different authentication mechanisms, request protocols, and response formats; "
                "please refer to the corresponding official documentation for implementation details. "
                "Note that using online text generation APIs may incur additional usage costs depending on the selected service provider and API plan."
            )

        label = self._open_label(label_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

        if self._label_transform:
            label = self._gt_transform(label) 

        if self.preprocess is not None:
            rgb, rgb_gt, label, x, x_gt, mask = self.preprocess(rgb, rgb_gt, label, x, x_gt)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            rgb_gt = torch.from_numpy(np.ascontiguousarray(rgb_gt)).float()
            label = torch.from_numpy(np.ascontiguousarray(label)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
            x_gt = torch.from_numpy(np.ascontiguousarray(x_gt)).float()
            if mask is not None:
                mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        output_dict = dict(rgb=rgb, rgb_gt=rgb_gt, label=label, modal_x=x, modal_x_gt=x_gt, text=text, mask=mask, name=str(name))
        return output_dict
    
    def _get_file_list(self, root_path):
        file_list = []
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    file_list.append(os.path.join(root, file))
        return file_list

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = cv2.imread(filepath)
        # Check if the image is grayscale (i.e., it has only one channel)
        if len(img.shape) == 2 or img.shape[2] == 1:
            # Convert grayscale image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, mode)

        img = np.array(img, dtype=dtype)
        return img
    
    @staticmethod
    def _open_text(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    
    @staticmethod
    def _open_label(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
