from .transforms import *
import os
from PIL import Image
import torchvision as tv


class TestDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split):
        self.root = root
        self.year = year
        self.split = split
        self.init_data()

    def read_img(self, path):
        pic = Image.open(path).convert('RGB')
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path).convert('P')
        transform = LabelToLongTensor()
        return transform(pic)

    def init_data(self):
        with open(os.path.join(self.root, 'ImageSets', self.year, self.split + '.txt'), 'r') as f:
            self.video_list = sorted(f.read().splitlines())
            print('--- DAVIS {} {} loaded for testing ---'.format(self.year, self.split))

    def get_snippet(self, video_name, frame_ids):
        img_path = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_path = os.path.join(self.root, 'Annotations', '480p', video_name)
        imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        given_masks = [self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(0))).unsqueeze(0)] + [None] * (len(frame_ids) - 1)
        files = ['{:05d}.png'.format(i) for i in frame_ids]
        if self.split == 'test-dev':
            return {'imgs': imgs, 'given_masks': given_masks, 'files': files, 'val_frame_ids': None}
        masks = torch.stack([self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))) for i in frame_ids]).unsqueeze(0)
        if self.year == '2016':
            masks = (masks != 0).long()
            given_masks[0] = (given_masks[0] != 0).long()
        return {'imgs': imgs, 'given_masks': given_masks, 'masks': masks, 'files': files, 'val_frame_ids': None}

    def get_video(self, video_name):
        frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'JPEGImages', '480p', video_name))])
        yield self.get_snippet(video_name, frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
