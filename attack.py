import os
import sys
sys.path.append('./model')

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import random
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms as T
import numpy as np
import matplotlib.pyplot as plt

import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    AmbientLights,
    DirectionalLights,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)
import math

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import fnmatch
import wandb

from util.tools import tis, gtis, save_visualize_boxes, get_material_patch
from model.detect import (
    DifferenciableHumanDetector, ECCV22EarlyFusionDetector, ECCV22MiddleFusionDetector,
    YOLOvXDetector, ECCV22LateFusionDetector
)
from tqdm import tqdm
from datetime import datetime

import argparse
from color_mapping.color import load_color_mapper
import util.pytorch3d_modify as p3dmd
from util.tps import TPSGridGen
from util.tools import random_hsv_augmentation
import util.mesh as MU
import itertools

from typing import List

CURRENT_TIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class FLIRDataset(Dataset):
    def __init__(self, img_dir, img_h, img_w, mode):
        self.img_dir = img_dir
        self.imgsize = (img_w, img_h)
        self.mode = mode
        self.img_names = self._get_image_names()
        self.len = len(self.img_names)

    def _get_image_names(self):
        extensions = ['*.png', '*.jpg', '*.jpeg']
        img_names = []
        for ext in extensions:
            img_names.extend(fnmatch.filter(os.listdir(self.img_dir), ext))
        img_names.sort()
        return img_names

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx < len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert(self.mode)
        image = self.pad_and_scale(image)
        return T.ToTensor()(image)

    def pad_and_scale(self, img):
        w, h = img.size
        target_w, target_h = self.imgsize
        
        if target_w / w < target_h / h:
            new_h = int(w * target_h / target_w)
            padded_img = Image.new(self.mode, (w, new_h), (0, 0, 0) if self.mode == 'RGB' else 0)
            padded_img.paste(img, (0, (new_h - h) // 2))
        else:
            new_w = int(h * target_w / target_h)
            padded_img = Image.new(self.mode, (new_w, h), (0, 0, 0) if self.mode == 'RGB' else 0)
            padded_img.paste(img, ((new_w - w) // 2, 0))

        return T.Resize((target_h, target_w))(padded_img)

class JointFLIRDataset(Dataset):
    def __init__(self, rgb_dataset, thermal_dataset):
        assert len(rgb_dataset) == len(thermal_dataset), 'RGB and thermal dataset size mismatch'
        self.rgb_dataset = rgb_dataset
        self.thermal_dataset = thermal_dataset

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, idx):
        return self.rgb_dataset[idx], self.thermal_dataset[idx]

class JointFLIRDataloader:
    def __init__(self, rgb_img_dir, thermal_img_dir, img_h, img_w, batch_size, shuffle=True):
        self.rgb_dataset = FLIRDataset(rgb_img_dir, img_h, img_w, mode='RGB')
        self.thermal_dataset = FLIRDataset(thermal_img_dir, img_h, img_w, mode='L')
        self.joint_dataset = JointFLIRDataset(self.rgb_dataset, self.thermal_dataset)
        self.dataloader = DataLoader(
            self.joint_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

class JointPatch(nn.Module):
    
    def __init__(self, cfg_patch):
        super(JointPatch, self).__init__()
        self.cfg_patch = cfg_patch
        self.tshirt_probs = nn.Parameter(self._init_prob_tensor((cfg_patch.tshirt.num_grid_h, cfg_patch.tshirt.num_grid_w)))
        self.trousers_probs = nn.Parameter(self._init_prob_tensor((cfg_patch.trousers.num_grid_h, cfg_patch.trousers.num_grid_w)))
        self.tshirt_color_origs = nn.Parameter(self._init_color_tensor((3, cfg_patch.tshirt.num_grid_h, cfg_patch.tshirt.num_grid_w)))
        self.trousers_color_origs = nn.Parameter(self._init_color_tensor((3, cfg_patch.trousers.num_grid_h, cfg_patch.trousers.num_grid_w)))

    def _init_prob_tensor(self, shape):
        return torch.rand(shape) * 0.02 + 0.49

    def _init_color_tensor(self, shape):
        return torch.randn(shape)
        
    def forward(self):
        return (self.tshirt_probs, 
                self.trousers_probs, 
                torch.sigmoid(self.tshirt_color_origs), 
                torch.sigmoid(self.trousers_color_origs))
    
    def sample_soft(self):
        tshirt_probs, trousers_probs, tshirt_colors, trousers_colors = self.forward()
        tshirt_colors = self.cropped_scale_2d(tshirt_colors, self.cfg_patch.tshirt.cell_h, self.cfg_patch.tshirt.cell_w, self.cfg_patch.tshirt.res_h, self.cfg_patch.tshirt.res_w)
        trousers_colors = self.cropped_scale_2d(trousers_colors, self.cfg_patch.trousers.cell_h, self.cfg_patch.trousers.cell_w, self.cfg_patch.trousers.res_h, self.cfg_patch.trousers.res_w)
        tshirt_probs = self.cropped_scale_2d(tshirt_probs, self.cfg_patch.tshirt.cell_h, self.cfg_patch.tshirt.cell_w, self.cfg_patch.tshirt.res_h, self.cfg_patch.tshirt.res_w)
        trousers_probs = self.cropped_scale_2d(trousers_probs, self.cfg_patch.trousers.cell_h, self.cfg_patch.trousers.cell_w, self.cfg_patch.trousers.res_h, self.cfg_patch.trousers.res_w)
        return tshirt_probs, trousers_probs, tshirt_colors, trousers_colors
        
    def sample_hard(self):
        tshirt_probs, trousers_probs, tshirt_colors, trousers_colors = self.forward()
        tshirt_probs = (tshirt_probs > 0.5).float()
        trousers_probs = (trousers_probs > 0.5).float()
        tshirt_colors = self.cropped_scale_2d(tshirt_colors, self.cfg_patch.tshirt.cell_h, self.cfg_patch.tshirt.cell_w, self.cfg_patch.tshirt.res_h, self.cfg_patch.tshirt.res_w)
        trousers_colors = self.cropped_scale_2d(trousers_colors, self.cfg_patch.trousers.cell_h, self.cfg_patch.trousers.cell_w, self.cfg_patch.trousers.res_h, self.cfg_patch.trousers.res_w)
        tshirt_probs = self.cropped_scale_2d(tshirt_probs, self.cfg_patch.tshirt.cell_h, self.cfg_patch.tshirt.cell_w, self.cfg_patch.tshirt.res_h, self.cfg_patch.tshirt.res_w)
        trousers_probs = self.cropped_scale_2d(trousers_probs, self.cfg_patch.trousers.cell_h, self.cfg_patch.trousers.cell_w, self.cfg_patch.trousers.res_h, self.cfg_patch.trousers.res_w)
        return tshirt_probs, trousers_probs, tshirt_colors, trousers_colors
    
    def sample_soft_with_random_cancel(self, cancel_prob=0.0):
        tshirt_probs, trousers_probs, tshirt_colors, trousers_colors = self.forward()
        tshirt_mask = (torch.rand_like(tshirt_probs) < cancel_prob)
        trousers_mask = (torch.rand_like(trousers_probs) < cancel_prob)
        tshirt_probs = torch.where(tshirt_mask, (tshirt_probs > 0.5).float(), tshirt_probs)
        trousers_probs = torch.where(trousers_mask, (trousers_probs > 0.5).float(), trousers_probs)
        tshirt_colors = self.cropped_scale_2d(tshirt_colors, self.cfg_patch.tshirt.cell_h, self.cfg_patch.tshirt.cell_w, self.cfg_patch.tshirt.res_h, self.cfg_patch.tshirt.res_w)
        trousers_colors = self.cropped_scale_2d(trousers_colors, self.cfg_patch.trousers.cell_h, self.cfg_patch.trousers.cell_w, self.cfg_patch.trousers.res_h, self.cfg_patch.trousers.res_w)
        tshirt_probs = self.cropped_scale_2d(tshirt_probs, self.cfg_patch.tshirt.cell_h, self.cfg_patch.tshirt.cell_w, self.cfg_patch.tshirt.res_h, self.cfg_patch.tshirt.res_w)
        trousers_probs = self.cropped_scale_2d(trousers_probs, self.cfg_patch.trousers.cell_h, self.cfg_patch.trousers.cell_w, self.cfg_patch.trousers.res_h, self.cfg_patch.trousers.res_w)
        return tshirt_probs, trousers_probs, tshirt_colors, trousers_colors, tshirt_mask, trousers_mask
    
    def cropped_scale_2d(self, x, scale_h, scale_w, res_h, res_w):
        x = x.repeat_interleave(scale_h, dim = -2).repeat_interleave(scale_w, dim = -1)
        x = x[..., :res_h, :res_w]
        return x
    
    def binary_loss(self):
        tshirt_probs, trousers_probs, _, _ = self.forward()
        tshirt_binary_loss = (tshirt_probs * (1 - tshirt_probs)).mean()
        trousers_binary_loss = (trousers_probs * (1 - trousers_probs)).mean()
        return tshirt_binary_loss + trousers_binary_loss

class Simulator:
    def __init__(self, cfg_simulator: EasyDict):
        self.cfg_simulator = cfg_simulator
        self.image_size = 1500
        self.up = (0, 1, 0)
        self.fov = 45
        
        self.mesh_man = load_objs_as_meshes([self.cfg_simulator.man.obj_path], device='cuda')
        self.mesh_tshirt = load_objs_as_meshes([self.cfg_simulator.tshirt.obj_path], device='cuda')
        self.mesh_trouser = load_objs_as_meshes([self.cfg_simulator.trousers.obj_path], device='cuda')
        
        self.uv_man_rgb = torch.from_numpy(plt.imread(self.cfg_simulator.man.uv_rgb_path)).unsqueeze(0).cuda()
        self.uv_man_thermal = torch.from_numpy(plt.imread(self.cfg_simulator.man.uv_thermal_path)).unsqueeze(-1).expand(-1, -1, 3).unsqueeze(0).cuda()

        self.uv_clean_tshirt_thermal_list = [Image.open(os.path.join(self.cfg_simulator.tshirt.uv_thermal_dir, f)).convert('RGB') for f in os.listdir(self.cfg_simulator.tshirt.uv_thermal_dir)]
        self.uv_clean_trousers_thermal_list = [Image.open(os.path.join(self.cfg_simulator.trousers.uv_thermal_dir, f)).convert('RGB') for f in os.listdir(self.cfg_simulator.trousers.uv_thermal_dir)]
        self.uv_clean_tshirt_thermal_list = [T.ToTensor()(img).mean(dim=0).unsqueeze(0) for img in self.uv_clean_tshirt_thermal_list]
        self.uv_clean_trousers_thermal_list = [T.ToTensor()(img).mean(dim=0).unsqueeze(0) for img in self.uv_clean_trousers_thermal_list]
        
        self.color_mapper = load_color_mapper().cuda()
        
        self.tshirt_h = self.uv_clean_tshirt_thermal_list[0].shape[1]
        self.tshirt_w = self.uv_clean_tshirt_thermal_list[0].shape[2]
        self.trousers_h = self.uv_clean_trousers_thermal_list[0].shape[1]
        self.trousers_w = self.uv_clean_trousers_thermal_list[0].shape[2]
        
        self.initialize_tps_enhance()
        
    def initialize_tps_enhance(self):
        tshirt_num_points = (math.ceil(self.tshirt_h / self.cfg_simulator.tps_enhance.point_step) + 1, math.ceil(self.tshirt_w / self.cfg_simulator.tps_enhance.point_step) + 1)
        trousers_num_points = (math.ceil(self.trousers_h / self.cfg_simulator.tps_enhance.point_step) + 1, math.ceil(self.trousers_w / self.cfg_simulator.tps_enhance.point_step) + 1)
        tshirt_x_points = torch.linspace(-1.0, 1.0, tshirt_num_points[1])
        tshirt_y_points = torch.linspace(-1.0, 1.0, tshirt_num_points[0])
        trousers_x_points = torch.linspace(-1.0, 1.0, trousers_num_points[1])
        trousers_y_points = torch.linspace(-1.0, 1.0, trousers_num_points[0])
        tshirt_control_points = torch.Tensor(list(itertools.product(tshirt_x_points, tshirt_y_points)))
        trousers_control_points = torch.Tensor(list(itertools.product(trousers_x_points, trousers_y_points)))
        self.tps_enhance_tshirt = TPSGridGen(target_shape=torch.Size([self.tshirt_h, self.tshirt_w]), 
                                       target_control_points=tshirt_control_points,
                                       device='cuda')
        self.tshirt_num_points = tshirt_control_points.shape[0]
        self.tps_enhance_trousers = TPSGridGen(target_shape=torch.Size([self.trousers_h, self.trousers_w]), 
                                         target_control_points=trousers_control_points,
                                         device='cuda')
        self.trousers_num_points = trousers_control_points.shape[0]
        
        tshirt_point_mask = torch.ones(1, self.tshirt_num_points, 2)

        trousers_point_mask = torch.ones(1, self.trousers_num_points, 2)
        
        self.tshirt_point_mask = tshirt_point_mask
        self.trousers_point_mask = trousers_point_mask

    def apply_tps_enhance(self, tensors: List[torch.Tensor], mode = 'tshirt'):
        
        if mode == 'tshirt':
            tps = self.tps_enhance_tshirt
            h = self.tshirt_h
            w = self.tshirt_w
            num_points = self.tshirt_num_points
            point_mask = self.tshirt_point_mask
        elif mode == 'trousers':
            tps = self.tps_enhance_trousers
            h = self.trousers_h
            w = self.trousers_w
            num_points = self.trousers_num_points
            point_mask = self.trousers_point_mask
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
        random_offset = torch.rand(1, num_points, 2) * 2 - 1
        random_offset[:, :, 0] = random_offset[:, :, 0] * self.cfg_simulator.tps_enhance.max_pixels / h
        random_offset[:, :, 1] = random_offset[:, :, 1] * self.cfg_simulator.tps_enhance.max_pixels / w
        
        random_offset = random_offset * point_mask
        random_offset = random_offset
                
        transformed_tensors = []
        for tensor in tensors:
            transformed_tensor = tps.tps_trans(tensor.unsqueeze(0), random_offset=random_offset)
            transformed_tensors.append(transformed_tensor[0].squeeze(0))
        
        return tuple(transformed_tensors)
    
    def simulate(self, tshirt_probs, trousers_probs, tshirt_colors, trousers_colors, background_rgb, background_thermal):        
        tshirt_probs = tshirt_probs.cuda()
        trousers_probs = trousers_probs.cuda()   
        tshirt_colors = tshirt_colors.cuda()
        trousers_colors = trousers_colors.cuda()
        batch_size = background_rgb.shape[0]
        
        if self.cfg_simulator.tps_enhance.enable:
            tshirt_colors, tshirt_probs = self.apply_tps_enhance([tshirt_colors, tshirt_probs.unsqueeze(0)], mode='tshirt')
            trousers_colors, trousers_probs = self.apply_tps_enhance([trousers_colors, trousers_probs.unsqueeze(0)], mode='trousers')
            tshirt_probs = tshirt_probs.squeeze(0)
            trousers_probs = trousers_probs.squeeze(0)
            
        render_state = self.sample_render_state(batch_size)
        
        mesh_total_rgb = self.prepare_rgb_mesh(tshirt_probs, tshirt_colors, trousers_probs, trousers_colors)
                
        rendered_rgb, lab_man, _ = self.mesh_put(background_rgb.cuda(), mesh_total_rgb, 
                                                 pos=(render_state['pos_x'], render_state['pos_y']),
                                                 scale=render_state['scale'],
                                                 dist=render_state['dist'],
                                                 theta=render_state['theta'],
                                                 elev=render_state['elev'],
                                                 lights=render_state['lights_rgb'])
        
        mesh_total_thermal = self.prepare_thermal_mesh(tshirt_probs, trousers_probs)
        
        background_thermal = background_thermal.cuda().expand(-1, 3, -1, -1)
        rendered_thermal, lab_man_2, _ = self.mesh_put(background_thermal, mesh_total_thermal, 
                                                       pos=(render_state['pos_x'], render_state['pos_y']),
                                                       scale=render_state['scale'],
                                                       dist=render_state['dist'],
                                                       theta=render_state['theta'],
                                                       elev=render_state['elev'],
                                                       lights=render_state['lights_thermal'])
        
        assert torch.equal(lab_man, lab_man_2), 'render label mismatch'
        
        rendered_thermal = torch.clamp(rendered_thermal[:, [0], ...], min=0, max=1)
        rendered_rgb = torch.clamp(rendered_rgb, min=0, max=1)
        
        return rendered_rgb, rendered_thermal, lab_man

    def preprocess_uv_thermal(self, tshirt_probs, trousers_probs):
        rand_tshirt_clean = random.choice(self.uv_clean_tshirt_thermal_list).cuda()
        rand_trouser_clean = random.choice(self.uv_clean_trousers_thermal_list).cuda()
        rand_bg_tshirt = (torch.rand_like(tshirt_probs) * 0.36 + 0.04)
        rand_bg_trouser = (torch.rand_like(trousers_probs) * 0.36 + 0.04)
        tshirt_probs = tshirt_probs.unsqueeze(0) * rand_tshirt_clean + rand_bg_tshirt.unsqueeze(0) * (1 - tshirt_probs.unsqueeze(0))
        trousers_probs = trousers_probs.unsqueeze(0) * rand_trouser_clean + rand_bg_trouser.unsqueeze(0) * (1 - trousers_probs.unsqueeze(0))
        uv_tshirt_thermal = tshirt_probs.expand(3, -1, -1)
        uv_trouser_thermal = trousers_probs.expand(3, -1, -1)
        return uv_tshirt_thermal, uv_trouser_thermal
    
    def preprocess_uv_rgb(self, tshirt_probs, tshirt_colors, trousers_probs, trousers_colors):
        shirt_h, shirt_w = tshirt_colors.shape[-2:]
        trousers_h, trousers_w = trousers_colors.shape[-2:]
        if self.cfg_simulator.material.path is not None:
            grey_tshirt = get_material_patch(self.cfg_simulator.material.path, shirt_h, shirt_w, grid_size=1, kernel_size=1, sigma=5).cuda()
            grey_trouser = get_material_patch(self.cfg_simulator.material.path, trousers_h, trousers_w, grid_size=1, kernel_size=1, sigma=5).cuda()
        else:
            grey_tshirt = (torch.rand_like(tshirt_colors) * 0.24 + 0.64).cuda()
            grey_trouser = (torch.rand_like(trousers_colors) * 0.24 + 0.64).cuda()     
        uv_tshirt_rgb = tshirt_colors * tshirt_probs + grey_tshirt * (1 - tshirt_probs)
        uv_trouser_rgb = trousers_colors * trousers_probs + grey_trouser * (1 - trousers_probs)
        if self.cfg_simulator.color_mapping:
            uv_tshirt_rgb = torch.clamp(uv_tshirt_rgb, 0, 1)
            uv_trouser_rgb = torch.clamp(uv_trouser_rgb, 0, 1)
            uv_tshirt_rgb = self.color_mapper.auto_color_mapping(uv_tshirt_rgb)
            uv_trouser_rgb = self.color_mapper.auto_color_mapping(uv_trouser_rgb)
        return uv_tshirt_rgb, uv_trouser_rgb
        
    def sample_render_state(self, batch_size):
        render_config = self.cfg_simulator.render
        
        dist = torch.rand(batch_size).cuda() * (render_config.dist[1] - render_config.dist[0]) + render_config.dist[0]
        theta = torch.rand(batch_size).cuda() * (render_config.theta[1] - render_config.theta[0]) + render_config.theta[0]
        pos_x = torch.rand(batch_size).cuda() * (render_config.pos_x[1] - render_config.pos_x[0]) + render_config.pos_x[0]
        pos_y = torch.rand(batch_size).cuda() * (render_config.pos_y[1] - render_config.pos_y[0]) + render_config.pos_y[0]
        scale = torch.rand(batch_size).cuda() * (render_config.scale[1] - render_config.scale[0]) + render_config.scale[0]
        elev = torch.rand(batch_size).cuda() * (render_config.elev[1] - render_config.elev[0]) + render_config.elev[0]
        
        lights_rgb = self.sample_lights_rgb(batch_size, render_config.light)
        lights_thermal = self.sample_lights_thermal(batch_size, render_config.light)
        
        return {
            'dist': dist,
            'theta': theta,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'scale': scale,
            'elev': elev,
            'lights_rgb': lights_rgb,
            'lights_thermal': lights_thermal
        }
        
    def sample_lights_rgb(self, batch_size, light_type = None):
        lights = [AmbientLights(device='cuda',
                              ambient_color=[[0.9, 0.88, 0.85]])
                 for _ in range(batch_size)]
        return lights

    def sample_lights_thermal(self, batch_size, light_type = None):
        lights = [AmbientLights(device='cuda',
                              ambient_color=[[0.9, 0.88, 0.85]])
                 for _ in range(batch_size)]
        return lights

    def prepare_rgb_mesh(self, tshirt_probs, tshirt_colors, trousers_probs, trousers_colors):
        
        uv_tshirt_rgb, uv_trouser_rgb = self.preprocess_uv_rgb(tshirt_probs, tshirt_colors, trousers_probs, trousers_colors)
        
        textures_man_rgb = TexturesUV(maps=self.uv_man_rgb,
                                      faces_uvs=self.mesh_man.textures.faces_uvs_list(),
                                      verts_uvs=self.mesh_man.textures.verts_uvs_list())
        textures_tshirt_rgb = TexturesUV(maps=uv_tshirt_rgb.unsqueeze(0).permute(0, 2, 3, 1).cuda(),
                                         faces_uvs=self.mesh_tshirt.textures.faces_uvs_list(),
                                         verts_uvs=self.mesh_tshirt.textures.verts_uvs_list())
        textures_trouser_rgb = TexturesUV(maps=uv_trouser_rgb.unsqueeze(0).permute(0, 2, 3, 1).cuda(),
                                          faces_uvs=self.mesh_trouser.textures.faces_uvs_list(),
                                          verts_uvs=self.mesh_trouser.textures.verts_uvs_list())
        
        self.mesh_man.textures = textures_man_rgb
        self.mesh_tshirt.textures = textures_tshirt_rgb
        self.mesh_trouser.textures = textures_trouser_rgb
        
        return self.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])

    def prepare_thermal_mesh(self, tshirt_probs, trousers_probs):
        
        uv_tshirt_thermal, uv_trouser_thermal = self.preprocess_uv_thermal(tshirt_probs, trousers_probs)
        
        textures_man_thermal = TexturesUV(maps=self.uv_man_thermal,
                                          faces_uvs=self.mesh_man.textures.faces_uvs_list(),
                                          verts_uvs=self.mesh_man.textures.verts_uvs_list())
        textures_tshirt_thermal = TexturesUV(maps=uv_tshirt_thermal.unsqueeze(0).permute(0, 2, 3, 1).cuda(),
                                             faces_uvs=self.mesh_tshirt.textures.faces_uvs_list(),
                                             verts_uvs=self.mesh_tshirt.textures.verts_uvs_list())
        textures_trouser_thermal = TexturesUV(maps=uv_trouser_thermal.unsqueeze(0).permute(0, 2, 3, 1).cuda(),
                                              faces_uvs=self.mesh_trouser.textures.faces_uvs_list(),
                                              verts_uvs=self.mesh_trouser.textures.verts_uvs_list())
        
        self.mesh_man.textures = textures_man_thermal
        self.mesh_tshirt.textures = textures_tshirt_thermal
        self.mesh_trouser.textures = textures_trouser_thermal
        
        return self.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])

    def view_mesh(self, mesh, camera_loc, lights, device=None):
        if device is None:
            device = mesh.device

        R, T = look_at_view_transform(device=device, *camera_loc, up=(self.up,))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=self.fov)

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=p3dmd.AdaSafeSoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

        images = renderer(mesh, lights=lights, cameras=cameras)
        return images

    def mesh_put(self, x, mesh, pos, scale, theta, dist, elev, lights):
        B, C, H, W = x.shape
        scale = scale[:B]
        dist = dist[:B]
        elev = elev[:B]
        theta = theta[:B]
        lights = lights[:B]
        pos = (pos[0][:B], pos[1][:B])

        images = []
        for i in range(len(dist)):
            image = self.view_mesh(mesh, (dist[i], elev[i], theta[i]), lights=lights[i])
            images.append(image)
        
        images = torch.cat(images, 0)
        images = images.permute(0, 3, 1, 2)

        images[:, -1, ...] = (images[:, -1, ...] > 0).to(images)

        theta = x.new_zeros(B, 2, 3)
        theta[:, 0, 0] = scale
        theta[:, 0, 1] = 0
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = pos[0]
        theta[:, 1, 2] = pos[1]

        grid = F.affine_grid(theta, x.shape, align_corners=True)
        images = F.grid_sample(images, grid, padding_mode='zeros', align_corners=True)

        masks = images[:, -1:, ...]
        images = images[:, :3, ...]

        x_new = images * masks + x * (1 - masks)

        lab_new = x.new(size=[B, 1, 5])
        lab_new[:, 0, 0] = 0.0

        mesh_bord = [torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in masks]
        mesh_bord = torch.stack(mesh_bord)

        lab_new[:, 0, 2] = mesh_bord[:, 0]
        lab_new[:, 0, 1] = mesh_bord[:, 1]
        lab_new[:, 0, 4] = mesh_bord[:, 2]
        lab_new[:, 0, 3] = mesh_bord[:, 3]

        return x_new, lab_new, images

    def join_meshes(self, meshes, join_maps=None):
        verts = []
        faces = []
        verts_uvs = []
        faces_uvs = []
        maps = []

        for mesh in meshes:
            verts.append(mesh.verts_packed())
            faces.append(mesh.faces_packed())
            maps.append(mesh.textures.maps_list()[0])
            verts_uvs.append(mesh.textures.verts_uvs_list()[0])
            faces_uvs.append(mesh.textures.faces_uvs_list()[0])

        w = 0
        h = 0
        pos = []
        for m in maps:
            if m.shape[0] > w:
                w = m.shape[0]
            h = h + m.shape[1]

        hi = 0
        v_num = 0
        vuv_num = 0
        for i in range(len(meshes)):
            verts_uvs[i] = torch.stack(
                [(verts_uvs[i][:, 0] * maps[i].shape[1] + hi) / h, verts_uvs[i][:, 1] * maps[i].shape[0] / w], -1)
            hi = hi + maps[i].shape[1]

            faces[i] = faces[i] + v_num
            v_num += len(verts[i])

            faces_uvs[i] = faces_uvs[i] + vuv_num
            vuv_num += len(verts_uvs[i])

        if join_maps is None:
            maps = [F.pad(m, (0, 0, 0, 0, w - m.shape[0], 0)) for m in maps]
            join_maps = [torch.cat(maps, 1)]

        verts = [torch.cat(verts)]
        faces = [torch.cat(faces)]
        verts_uvs = [torch.cat(verts_uvs)]
        faces_uvs = [torch.cat(faces_uvs)]

        textures = pytorch3d.renderer.mesh.textures.TexturesUV(join_maps, faces_uvs, verts_uvs)

        return Meshes(verts=verts, faces=faces, textures=textures)

def load_detector(config_detector, training):
    if 'joint' in config_detector:
        if config_detector.joint.name == 'eccv22earlyfusion':
            return ECCV22EarlyFusionDetector(training=training)
        elif config_detector.joint.name == 'eccv22middlefusion':
            return ECCV22MiddleFusionDetector(training=training)
        elif config_detector.joint.name == 'eccv22latefusion':
            return ECCV22LateFusionDetector(training=training)
        elif config_detector.joint.name.startswith('yolo11'):
            return YOLOvXDetector(training=training, version=config_detector.joint.name, mode='jointmax')
    raise ValueError(f'Unsupported detector in minimal repo: {config_detector}')

def run_validation(config, patch, detector, val_dataloader, simulator, step):
    with torch.no_grad():

        val_confidence_soft = 0
        val_confidence_hard = 0
        num_val_samples = min(config.train.num_val_samples, len(val_dataloader)) if hasattr(config.train, 'num_val_samples') else len(val_dataloader)

        successful_attacks_soft = 0
        successful_attacks_hard = 0

        for idx, (background_rgb, background_thermal) in enumerate(val_dataloader):
            if idx >= num_val_samples:
                break

            tshirt_probs, trousers_probs, tshirt_colors, trousers_colors = patch.sample_soft()
            rendered_rgb, rendered_thermal, lab_man = simulator.simulate(
                tshirt_probs,
                trousers_probs,
                tshirt_colors,
                trousers_colors,
                background_rgb,
                background_thermal
            )
            label_boxes = lab_man.squeeze(1)[:, 1:]
            confidence = detector.boxed_detect(rendered_rgb, rendered_thermal, label_boxes, config.train.iou_thr)
            val_confidence_soft += confidence.item()
            if confidence.item() < config.train.conf_thr:
                successful_attacks_soft += 1
                
            tshirt_probs, trousers_probs, tshirt_colors, trousers_colors = patch.sample_hard()
            rendered_rgb, rendered_thermal, lab_man = simulator.simulate(
                tshirt_probs,
                trousers_probs,
                tshirt_colors,
                trousers_colors,
                background_rgb,
                background_thermal
            )
            label_boxes = lab_man.squeeze(1)[:, 1:]
            confidence = detector.boxed_detect(rendered_rgb, rendered_thermal, label_boxes, config.train.iou_thr)
            val_confidence_hard += confidence.item()
            if confidence.item() < config.train.conf_thr:
                successful_attacks_hard += 1

        val_avg_confidence_soft = val_confidence_soft / num_val_samples
        val_avg_confidence_hard = val_confidence_hard / num_val_samples
        ASR_soft = successful_attacks_soft / num_val_samples
        ASR_hard = successful_attacks_hard / num_val_samples

        wandb.log({
            'val_confidence_soft': val_avg_confidence_soft,
            'val_confidence_hard': val_avg_confidence_hard,
            'ASR_soft': ASR_soft,
            'ASR_hard': ASR_hard
        }, step=step)
        print(f'Validation @ step {step}:')
        print(f'Val Confidence Soft = {val_avg_confidence_soft:.4f}, Val Confidence Hard = {val_avg_confidence_hard:.4f}')
        print(f'ASR Soft = {ASR_soft:.4f}, ASR Hard = {ASR_hard:.4f}')

        return {
            'val_confidence_soft': val_avg_confidence_soft,
            'val_confidence_hard': val_avg_confidence_hard,
            'ASR_soft': ASR_soft,
            'ASR_hard': ASR_hard
        }

def training_loop(config, patch, train_dataloader, val_dataloader, simulator):
        
    detector = load_detector(config.detector, training=True)
    
    trained_params = []
    for name, param in patch.named_parameters():
        trained_params.append(param)
    print(f'number of trained_params: {len(trained_params)}')
            
    optimizer = torch.optim.Adam(trained_params, lr=config.train.learning_rate)
    
    scheduler = None
    if config.train.ReduceLROnPlateau.min_lr is not None and config.train.ReduceLROnPlateau.lr_patience is not None and config.train.ReduceLROnPlateau.lr_factor is not None:
        print('Using ReduceLROnPlateau')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config.train.ReduceLROnPlateau.lr_patience, factor=config.train.ReduceLROnPlateau.lr_factor)
    elif config.train.CosineAnnealingWarmRestarts.T_0 is not None and config.train.CosineAnnealingWarmRestarts.T_mult is not None and config.train.CosineAnnealingWarmRestarts.eta_min is not None:
        print('Using CosineAnnealingWarmRestarts')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.train.CosineAnnealingWarmRestarts.T_0, T_mult=config.train.CosineAnnealingWarmRestarts.T_mult, eta_min=config.train.CosineAnnealingWarmRestarts.eta_min)
    
    if scheduler is None:
        print('Using Constant LR')
        
    if config.train.save_dir is not None:
        config.train.save_dir = os.path.join(config.train.save_dir, CURRENT_TIME_STRING)
        os.makedirs(config.train.save_dir, exist_ok=True)
        with open(os.path.join(config.train.save_dir, 'settings.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    if config.train.load_ckpt is not None:
        patch.load_state_dict(torch.load(config.train.load_ckpt)['patch_state_dict'], strict=False)
        
    if config.train.save_dir is not None and config.train.save_interval is not None:
        ckpt_path = os.path.join(config.train.save_dir, f'epoch-0000.pt')
        torch.save({
            'epoch': 0,
            'patch_state_dict': patch.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
    
    val_interval = getattr(config.train, 'val_interval', None)
    eval_at_start = getattr(config.train, 'eval_at_start', False)

    if eval_at_start:
        run_validation(config, patch, detector, val_dataloader, simulator, step=0)

    for epoch in range(1, config.train.max_epoch + 1):
        
        epoch_avg_loss = 0
        
        for idx, (background_rgb, background_thermal) in tqdm(enumerate(train_dataloader), 
                                                          desc=f'Epoch {epoch:04d}/{config.train.max_epoch:04d}',
                                                          total=len(train_dataloader)):
            optimizer.zero_grad()
            
            tshirt_probs, trousers_probs, tshirt_colors, trousers_colors, tshirts_mask, trousers_mask = patch.sample_soft_with_random_cancel(config.train.random_cancel_prob)
            tshirt_colors = random_hsv_augmentation(tshirt_colors)
            trousers_colors = random_hsv_augmentation(trousers_colors)
            rendered_rgb, rendered_thermal, lab_man = simulator.simulate(tshirt_probs, 
                                                                         trousers_probs, 
                                                                         tshirt_colors, 
                                                                         trousers_colors, 
                                                                         background_rgb, 
                                                                         background_thermal)
            label_boxes = lab_man.squeeze(1)[:, 1:]
            
            if torch.isnan(rendered_rgb).any() or torch.isnan(rendered_thermal).any():
                print(f'Warning: rendered_rgb or rendered_thermal is nan at epoch {epoch}, step {idx}')
                continue
            
            confidence = detector.boxed_detect(rendered_rgb, rendered_thermal, label_boxes, config.train.iou_thr)
            det_loss = confidence
            
            binary_loss = patch.binary_loss() * config.train.binary_loss_scale
            total_loss = det_loss + binary_loss
            
            if torch.isnan(total_loss):
                print(f'Warning: total_loss is nan at epoch {epoch}, step {idx}')
                continue
            
            try:
                total_loss.backward()
                epoch_avg_loss += total_loss.item()
            except:
                print(f'Warning: total_loss.backward() failed at epoch {epoch}, step {idx}')
                continue
            
            if config.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(patch.parameters(), config.train.grad_clip)
                
            with torch.no_grad():
                if tshirts_mask is not None:
                    patch.tshirt_probs.grad *= (~tshirts_mask).float()
                if trousers_mask is not None:
                    patch.trousers_probs.grad *= (~trousers_mask).float()
                
            optimizer.step()
            
            with torch.no_grad():
                patch.tshirt_probs.data.clamp_(0, 1)
                patch.trousers_probs.data.clamp_(0, 1)
            
            wandb.log({
                'total_loss': total_loss.item(),
                'det_loss': det_loss.item(),
                'binary_loss_w_scale': binary_loss.item(),
                'binary_loss_wo_scale': patch.binary_loss().item(),
                'confidence': confidence.item(),
                'lr': optimizer.param_groups[0]['lr']
            }, step=(epoch - 1) * len(train_dataloader) + idx + 1)

            global_step = (epoch - 1) * len(train_dataloader) + idx + 1
            if val_interval is not None and global_step % val_interval == 0:
                run_validation(config, patch, detector, val_dataloader, simulator, step=global_step)
        
        epoch_avg_loss /= len(train_dataloader)
        
        print(f'Epoch {epoch}: Train Loss = {epoch_avg_loss:.4f}')
        if val_interval is None:
            run_validation(config, patch, detector, val_dataloader, simulator, step=epoch * len(train_dataloader))
        
        if config.train.ReduceLROnPlateau.min_lr is not None and config.train.ReduceLROnPlateau.lr_patience is not None and config.train.ReduceLROnPlateau.lr_factor is not None:
            print('Updating ReduceLROnPlateau')
            scheduler.step(epoch_avg_loss)
        elif config.train.CosineAnnealingWarmRestarts.T_0 is not None and config.train.CosineAnnealingWarmRestarts.T_mult is not None and config.train.CosineAnnealingWarmRestarts.eta_min is not None:
            print('Updating CosineAnnealingWarmRestarts')
            scheduler.step()
        
        if (config.train.save_dir is not None and config.train.save_interval is not None and 
            epoch % config.train.save_interval == 0):
            
            ckpt_path = os.path.join(config.train.save_dir, f'epoch-{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                'patch_state_dict': patch.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            
            with torch.no_grad():
                tshirt_probs, trousers_probs, tshirt_colors, trousers_colors = patch.sample_hard()
                gtis(tshirt_probs.cpu(), os.path.join(config.train.save_dir, f'epoch-{epoch:04d}_tshirt_probs.png'))
                gtis(trousers_probs.cpu(), os.path.join(config.train.save_dir, f'epoch-{epoch:04d}_trousers_probs.png'))
                tis(tshirt_colors.cpu(), os.path.join(config.train.save_dir, f'epoch-{epoch:04d}_tshirt_colors.png'))
                tis(trousers_colors.cpu(), os.path.join(config.train.save_dir, f'epoch-{epoch:04d}_trousers_colors.png'))

                rgb_renders = []
                thermal_renders = []
                for _ in range(12):
                    background_rgb, background_thermal = next(iter(train_dataloader))
                    background_rgb = background_rgb[[0]]
                    background_thermal = background_thermal[[0]]
                    
                    rendered_rgb, rendered_thermal, _ = simulator.simulate(
                        tshirt_probs, trousers_probs, tshirt_colors, trousers_colors,
                        background_rgb, background_thermal
                    )
                    
                    rgb_renders.append(rendered_rgb[0].cpu().permute(1,2,0).numpy())
                    thermal_renders.append(rendered_thermal[0].cpu().permute(1,2,0).numpy())
                
                h, w = rgb_renders[0].shape[:2]
                rgb_grid = np.zeros((h * 3, w * 4, 3))
                thermal_grid = np.zeros((h * 3, w * 4, 3))
                
                for i in range(12):
                    row = i // 4
                    col = i % 4
                    rgb_grid[row*h:(row+1)*h, col*w:(col+1)*w] = rgb_renders[i]
                    thermal_grid[row*h:(row+1)*h, col*w:(col+1)*w] = thermal_renders[i]
                
                plt.imsave(os.path.join(config.train.save_dir, f'epoch-{epoch:04d}_rgb_renders.png'), rgb_grid)
                plt.imsave(os.path.join(config.train.save_dir, f'epoch-{epoch:04d}_thermal_renders.png'), thermal_grid)

def evaluate_loop(config, patch, dataloader, simulator):
    
    detector1 = load_detector(config.detector, training=False)
    detector2 = load_detector(config.detector, training=True)
    global_num = 0
    
    save_dir = os.path.join(config.demo.save_dir, CURRENT_TIME_STRING)
    config.demo.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if config.demo.load_ckpt is not None:
        patch.load_state_dict(torch.load(config.demo.load_ckpt)['patch_state_dict'], strict=False)
    
    alphas = np.linspace(0, 1, 11)
    results = {alpha: {'total_confidence': 0, 'count': 0} for alpha in alphas}
    
    for background_rgb, background_thermal in tqdm(dataloader, total=min(len(dataloader), config.demo.num_examples)):
        background_rgb = background_rgb[[0]]
        background_thermal = background_thermal[[0]]
        global_num += 1
        if global_num > config.demo.num_examples:
            break
            
        for alpha in alphas[::-1]:
            print(f'Evaluating alpha: {alpha:.2f}')
            tshirt_probs_hard, trousers_probs_hard, tshirt_colors_hard, trousers_colors_hard = patch.sample_hard()
            tshirt_probs_soft, trousers_probs_soft, tshirt_colors_soft, trousers_colors_soft = patch.sample_soft()
            
            tshirt_probs = alpha * tshirt_probs_hard + (1 - alpha) * tshirt_probs_soft
            trousers_probs = alpha * trousers_probs_hard + (1 - alpha) * trousers_probs_soft
            tshirt_colors = alpha * tshirt_colors_hard + (1 - alpha) * tshirt_colors_soft
            trousers_colors = alpha * trousers_colors_hard + (1 - alpha) * trousers_colors_soft
            
            rendered_rgb, rendered_thermal, lab_man = simulator.simulate(
                tshirt_probs, trousers_probs, tshirt_colors, trousers_colors,
                background_rgb, background_thermal
            )
            
            label_boxes = lab_man.squeeze(1)[:, 1:]
            boxes, scores = detector1.display_detect(rendered_rgb, rendered_thermal)
            max_conf = detector2.boxed_detect(rendered_rgb, rendered_thermal, label_boxes, config.train.iou_thr)
            
            results[alpha]['total_confidence'] += max_conf.item()
            results[alpha]['count'] += 1
            
            save_dir_alpha = os.path.join(save_dir, f'alpha_{alpha:.2f}')
            os.makedirs(save_dir_alpha, exist_ok=True)
            
            rgb_path = os.path.join(save_dir_alpha, f'{global_num:04d}_rgb.png')
            thermal_path = os.path.join(save_dir_alpha, f'{global_num:04d}_thermal.png')
            save_visualize_boxes(rendered_rgb[0], rendered_thermal[0], boxes[0], scores[0], rgb_path, thermal_path)
    
    avg_confidences = {}
    for alpha in alphas:
        avg_conf = results[alpha]['total_confidence'] / results[alpha]['count']
        avg_confidences[alpha] = avg_conf
        print(f'Alpha {alpha:.2f}: Average confidence = {avg_conf:.4f}')
    
    if 'metrics' not in config:
        config.metrics = EasyDict()
    config.metrics.interpolation_results = {str(k): v for k, v in avg_confidences.items()}
    
    with open(os.path.join(save_dir, 'settings.json'), 'w') as f:
        json.dump(config, f, indent=4)

def deep_update(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1

def recouncile_patch(cfg_patch):
    if cfg_patch.global_cell:
        cfg_patch.tshirt.cell_h = cfg_patch.global_cell
        cfg_patch.tshirt.cell_w = cfg_patch.global_cell
        cfg_patch.trousers.cell_h = cfg_patch.global_cell
        cfg_patch.trousers.cell_w = cfg_patch.global_cell
        cfg_patch.tshirt.num_grid_h = cfg_patch.tshirt.res_h // cfg_patch.global_cell if cfg_patch.tshirt.res_h % cfg_patch.global_cell == 0 else cfg_patch.tshirt.res_h // cfg_patch.global_cell + 1
        cfg_patch.tshirt.num_grid_w = cfg_patch.tshirt.res_w // cfg_patch.global_cell if cfg_patch.tshirt.res_w % cfg_patch.global_cell == 0 else cfg_patch.tshirt.res_w // cfg_patch.global_cell + 1
        cfg_patch.trousers.num_grid_h = cfg_patch.trousers.res_h // cfg_patch.global_cell if cfg_patch.trousers.res_h % cfg_patch.global_cell == 0 else cfg_patch.trousers.res_h // cfg_patch.global_cell + 1
        cfg_patch.trousers.num_grid_w = cfg_patch.trousers.res_w // cfg_patch.global_cell if cfg_patch.trousers.res_w % cfg_patch.global_cell == 0 else cfg_patch.trousers.res_w // cfg_patch.global_cell + 1
    return cfg_patch
        
def main():
    default_config = os.path.join(os.path.dirname(__file__), 'configs', 'default.json')
    default_config = json.load(open(default_config, 'r'))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=str, required=False)
    parser.add_argument('--load_from', type=str, required=False)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        configs = json.load(f)
    
    default_config = deep_update(default_config, configs)
    config = EasyDict(default_config)
    
    if args.mode is not None:
        config.mode = args.mode
    if args.load_from is not None:
        config.patch.load_from = args.load_from
    if args.no_wandb and args.wandb:
        raise ValueError('Cannot use --no_wandb and --wandb at the same time.')
    if args.no_wandb:
        config.wandb.mode = 'disabled'
    elif args.wandb:
        config.wandb.mode = 'online'
    print(f'Mode: {config.mode}')

    config.patch = recouncile_patch(config.patch)
    
    patch = JointPatch(config.patch)
    if config.patch.load_from is not None:
        load_result = patch.load_state_dict(torch.load(config.patch.load_from)['patch_state_dict'], strict=False)
        print(f'Loaded patch from {config.patch.load_from}')
        if load_result.missing_keys:
            print(f'Missing keys: {load_result.missing_keys}')
        if load_result.unexpected_keys:
            print(f'Unexpected keys: {load_result.unexpected_keys}')
    
    train_dataloader = JointFLIRDataloader(
        config.dataset.train.rgb_path,
        config.dataset.train.thermal_path,
        config.dataset.out_h,
        config.dataset.out_w,
        config.dataset.batch_size,
        shuffle=True
    )
    val_dataloader = JointFLIRDataloader(
        config.dataset.val.rgb_path,
        config.dataset.val.thermal_path,
        config.dataset.out_h,
        config.dataset.out_w,
        batch_size=1,
        shuffle=False
    )
    
    simulator = Simulator(config.simulator)
    
    if config.mode == 'train':
        wandb.init(project=config.wandb.project, 
           name=f'{config.wandb.name}_{CURRENT_TIME_STRING}', 
           mode=config.wandb.mode,
           config=config)
        training_loop(config, patch, train_dataloader, val_dataloader, simulator)
    elif config.mode == 'evaluate':  
        evaluate_loop(config, patch, val_dataloader, simulator)

if __name__ == "__main__":
    main()
