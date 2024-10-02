from torch.utils.data import Dataset
import numpy as np
import torch

def sg2img(start, goal, img_size=128):
    img = torch.zeros((2, img_size, img_size))
    img[0, start[0], start[1]] = 1
    img[1, goal[0], goal[1]] = 1
    return img.float()


class DemData(Dataset):
    def __init__(self, split='train'):
        data = np.load(split + '.npz')
        data_focal = np.load(split + '_focal.npz')
        self.dems = torch.tensor(data['dem']).float()
        self.starts = data_focal['start']
        self.goals = data_focal['goal']
        self.focal = torch.tensor(data_focal['focal']).float()
        self.rgb = torch.tensor(data['rgb']).float() / 255.
        self.img_size = self.rgb.shape[-1]
    
    def __len__(self):
        return len(self.dems) * 10
    
    def __getitem__(self, idx):
        map_idx, task_idx = idx // 10, idx % 10
        dem = self.dems[map_idx]
        dem = dem - dem.min()
        dem = dem / dem.max()
        rgb = self.rgb[map_idx]
        start = self.starts[map_idx][task_idx]
        goal = self.goals[map_idx][task_idx]
        focal = self.focal[map_idx][task_idx]
        sg = sg2img(start, goal, img_size=self.img_size)
        return dem, rgb, sg, focal
