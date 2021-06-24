import os
from os.path import join

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class CloDataSet(Dataset):
    def __init__(self, root_dir, 
                 split='train', sample_spacing=1, img_size=32, scan_npoints=-1):
        self.datadir = join(root_dir, split)
        self.split = split
        self.img_size = img_size
        self.spacing = sample_spacing
        self.scan_npoints = scan_npoints
        self.f = np.load(join(SCRIPT_DIR, '..', 'assets', 'smpl_faces.npy'))

        self.posmap, self.scan_n, self.scan_pc = [], [], []
        self.scan_name, self.body_verts, self.pose_params =  [], [], []
        self._init_dataset()
        self.data_size = int(len(self.posmap))

        print('Data loaded, in total {} {} examples.\n'.format(self.data_size, self.split))

    def _init_dataset(self):
        print('Loading {} data...'.format(self.split))
        flist = sorted(os.listdir(self.datadir))[::self.spacing]
        for fn in tqdm(flist):
            dd = np.load(join(self.datadir, fn)) # dd: 'data dict'

            self.posmap.append(torch.tensor(dd['posmap{}'.format(self.img_size)]).float().permute([2,0,1]))
            self.scan_name.append(str(dd['scan_name']))
            self.body_verts.append(torch.tensor(dd['body_verts']).float())
            self.scan_n.append(torch.tensor(dd['scan_n']).float())
            self.scan_pc.append(torch.tensor(dd['scan_pc']).float()) # scan_pc: the GT point cloud. 

    def __getitem__(self, index):
        posmap = self.posmap[index]
        scan_name = self.scan_name[index]
        body_verts = self.body_verts[index]

        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]

        if self.scan_npoints != -1: 
            selected_idx = torch.randperm(len(scan_n))[:self.scan_npoints]
            scan_pc = scan_pc[selected_idx, :]
            scan_n = scan_n[selected_idx, :]

        return posmap, scan_n, scan_pc, scan_name, body_verts, torch.tensor(index).long()

    def __len__(self):
        return self.data_size
