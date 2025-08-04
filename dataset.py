import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(img_path):
    with open(img_path, 'rb') as f:
            img =Image.open(f).convert('RGB')
    return img
class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform

        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))

    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        with open(img_path, 'rb') as f:
            img =Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        if self.split_type=='train':
            all_captions=[]
        else :
            matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
            all_captions = [self.captions[idx] for idx in matching_idxs]
            
        return torch.FloatTensor(img), torch.tensor(self.captions[index]), all_captions



    def __len__(self):
        return len(self.captions)




def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images = [item[0] for item in data]
    captions = [item[1] for item in data]
    all_captions = [item[2] for item in data]

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]  
    # targets 为填充到统一长度的caption
    
    return images, targets, lengths, all_captions 

