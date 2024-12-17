from .transforms import *
import os
from PIL import Image
import torchvision as tv


class TestYTVOS(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = root
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
        self.video_list = sorted(os.listdir(os.path.join(self.root, 'valid', 'Annotations')))
        print('--- YTVOS 2018 {} loaded for testing ---'.format(self.split))

    def get_snippet(self, video_name, frame_ids, val_frame_ids):
        img_path = os.path.join(self.root, 'valid_all_frames', 'JPEGImages', video_name)
        mask_path = os.path.join(self.root, 'valid', 'Annotations', video_name)
        imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        given_masks = [self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))).unsqueeze(0)
                       if i in sorted([int(file[:5]) for file in os.listdir(mask_path)]) else None for i in frame_ids]
        files = ['{:05d}.png'.format(i) for i in val_frame_ids]
        return {'imgs': imgs, 'given_masks': given_masks, 'files': files, 'val_frame_ids': val_frame_ids}

    def get_video(self, video_name):
        frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'valid_all_frames', 'JPEGImages', video_name))])
        val_frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'valid', 'JPEGImages', video_name))])
        min_frame_id = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'valid', 'Annotations', video_name))])[0]
        frame_ids = [i for i in frame_ids if i >= min_frame_id]
        val_frame_ids = [i for i in val_frame_ids if i >= min_frame_id]
        yield self.get_snippet(video_name, frame_ids, val_frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
