# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import math
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import open3d as open3d
import scipy
import skimage
import torch
import trimesh
from pytorch3d.structures import Pointclouds
from torch import Tensor

from home_robot.core.interfaces import Observations
from home_robot.motion import Footprint, PlanResult, RobotModel
from home_robot.utils.data_tools.dict import update
from home_robot.utils.morphology import binary_dilation, binary_erosion, get_edges
from home_robot.utils.point_cloud import (
    create_visualization_geometries,
    numpy_to_pcd,
    pcd_to_numpy,
    show_point_cloud,
)
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from home_robot.utils.visualization import create_disk
from home_robot.utils.voxel import VoxelizedPointcloud, scatter3d

import torch.nn.functional as F
import torchvision.transforms.functional as V
# from segment_anything import sam_model_registry, SamPredictor
# from transformers import AutoProcessor, OwlViTForObjectDetection
import clip
from torchvision import transforms
from home_robot.mapping.voxel.scannet import CLASS_LABELS_200
import os
# import wget
import cv2
import time

Frame = namedtuple(
    "Frame",
    [
        "camera_pose",
        "camera_K",
        "xyz",
        "rgb",
        "feats",
        "depth",
        "base_pose",
        "info",
        "full_world_xyz",
        "xyz_frame",
    ],
)

VALID_FRAMES = ["camera", "world"]

DEFAULT_GRID_SIZE = [512, 512]

logger = logging.getLogger(__name__)


def ensure_tensor(arr):
    if isinstance(arr, np.ndarray):
        return Tensor(arr)
    if not isinstance(arr, Tensor):
        raise ValueError(f"arr of unknown type ({type(arr)}) cannot be cast to Tensor")


class SparseVoxelMapV2(object):
    """Create a voxel map object which captures 3d information.

    This class represents a 3D voxel map used for capturing environmental information. It provides various parameters
    for configuring the map's properties, such as resolution, feature dimensions.

    Attributes:
        resolution (float): The size of a voxel in meters.
        feature_dim (int): The size of feature embeddings to capture per-voxel point.
        grid_size (Tuple[int, int]): The dimensions of the voxel grid (optional).
        grid_resolution (float): The resolution of the grid (optional).
        obs_min_height (float): The minimum height for observations.
        obs_max_height (float): The maximum height for observations.
        obs_min_density (float): The minimum density for observations.
        smooth_kernel_size (int): The size of the smoothing kernel.
        add_local_radius_points (bool): Whether to add local radius points.
        remove_visited_from_obstacles(bool): subtract out observations potentially of the robot
        local_radius (float): The radius for local points.
        min_depth (float): The minimum depth for observations.
        max_depth (float): The maximum depth for observations.
        pad_obstacles (int): Padding for obstacles.
        voxel_kwargs (Dict[str, Any]): Additional voxel configuration.
        map_2d_device (str): The device for 2D mapping.
    """

    DEFAULT_INSTANCE_MAP_KWARGS = dict(
        du_scale=1,
        instance_association="bbox_iou",
        log_dir_overwrite_ok=True,
        mask_cropped_instances="False",
    )

    def __init__(
        self,
        resolution: float = 0.01,
        feature_dim: int = 3,
        grid_size: Tuple[int, int] = None,
        grid_resolution: float = 0.05,
        obs_min_height: float = 0.1,
        obs_max_height: float = 1.8,
        obs_min_density: float = 10,
        smooth_kernel_size: int = 2,
        add_local_radius_points: bool = True,
        remove_visited_from_obstacles: bool = False,
        local_radius: float = 0.15,
        min_depth: float = 0.25,
        max_depth: float = 2.0,
        pad_obstacles: int = 0,
        voxel_kwargs: Dict[str, Any] = {},
        map_2d_device: str = "cpu",
        use_median_filter: bool = False,
        median_filter_size: int = 5,
        median_filter_max_error: float = 0.01,
        use_derivative_filter: bool = False,
        derivative_filter_threshold: float = 0.5,
        owl = False,
        device = 'cpu'
    ):
        """
        Args:
            resolution(float): in meters, size of a voxel
            feature_dim(int): size of feature embeddings to capture per-voxel point
        """
        print('------------------------YOU ARE NOW RUNNING PEIQI VOXEL MAP CODES V2-----------------')
        # TODO: We an use fastai.store_attr() to get rid of this boilerplate code
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.obs_min_height = obs_min_height
        self.obs_max_height = obs_max_height
        self.obs_min_density = obs_min_density

        # Smoothing kernel params
        self.smooth_kernel_size = smooth_kernel_size
        if self.smooth_kernel_size > 0:
            self.smooth_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(smooth_kernel_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.smooth_kernel = None

        # Median filter params
        self.median_filter_size = median_filter_size
        self.use_median_filter = use_median_filter
        self.median_filter_max_error = median_filter_max_error

        # Derivative filter params
        self.use_derivative_filter = use_derivative_filter
        self.derivative_filter_threshold = derivative_filter_threshold

        self.grid_resolution = grid_resolution
        self.voxel_resolution = resolution
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pad_obstacles = int(pad_obstacles)

        self.voxel_kwargs = voxel_kwargs
        self.map_2d_device = map_2d_device

        if self.pad_obstacles > 0:
            self.dilate_obstacles_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(self.pad_obstacles))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_obstacles_kernel = None

        # Add points with local_radius to the voxel map at (0,0,0) unless we receive lidar points
        self._add_local_radius_points = add_local_radius_points
        self._remove_visited_from_obstacles = remove_visited_from_obstacles
        self.local_radius = local_radius

        # Create disk for mapping explored areas near the robot - since camera can't always see it
        self._disk_size = np.ceil(self.local_radius / self.grid_resolution)

        self._visited_disk = torch.from_numpy(
            create_disk(self._disk_size, (2 * self._disk_size) + 1)
        ).to(map_2d_device)

        if grid_size is not None:
            self.grid_size = [grid_size[0], grid_size[1]]
        else:
            self.grid_size = DEFAULT_GRID_SIZE
        # Track the center of the grid - (0, 0) in our coordinate system
        # We then just need to update everything when we want to track obstacles
        self.grid_origin = Tensor(self.grid_size + [0], device=map_2d_device) // 2
        # Used to track the offset from our observations so maps dont use too much space

        # Used for tensorized bounds checks
        self._grid_size_t = Tensor(self.grid_size, device=map_2d_device)

        self.owl = owl
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device)
        self.clip_model.eval()
        if self.owl:
            self.owl_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
            self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval().to(device)
            if not os.path.exists('sam_vit_b_01ec64.pth'):
                wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', out = 'sam_vit_b_01ec64.pth')
            sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
            self.mask_predictor = SamPredictor(sam)
            self.mask_predictor.model = self.mask_predictor.model.eval().to(device)
            self.texts = [['a photo of ' + text for text in CLASS_LABELS_200]]

        # Init variables
        self.reset()

    def reset(self) -> None:
        """Clear out the entire voxel map."""
        self.observations = []
        # Create voxelized pointcloud
        self.voxel_pcd = VoxelizedPointcloud(
            voxel_size=self.voxel_resolution,
            dim_mins=None,
            dim_maxs=None,
            feature_pool_method="mean",
            **self.voxel_kwargs,
        )

        self._seq = 0
        self._2d_last_updated = -1
        # Create map here - just reset *some* variables
        self.reset_cache()

    def reset_cache(self):
        """Clear some tracked things"""
        # Stores points in 2d coords where robot has been
        self._visited = torch.zeros(self.grid_size, device=self.map_2d_device)

        self.voxel_pcd.reset()

        # Store 2d map information
        # This is computed from our various point clouds
        self._map2d = None

    def fix_type(self, tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert to tensor and float"""
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        return tensor.float()
    
    def forward_one_block(self, resblocks, x):
        q, k, v = None, None, None
        y = resblocks.ln_1(x)
        y = F.linear(y, resblocks.attn.in_proj_weight, resblocks.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C//3).permute(2, 0, 1, 3).reshape(3*N, L, C//3)
        y = F.linear(y, resblocks.attn.out_proj.weight, resblocks.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + resblocks.mlp(resblocks.ln_2(v))

        return v

    def extract_mask_clip_features(self, x, image_shape):
        with torch.no_grad():
            x = self.clip_model.visual.conv1(x)
            N, L, H, W = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            for idx in range(self.clip_model.visual.transformer.layers):
                if idx == self.clip_model.visual.transformer.layers - 1:
                    break
                x = self.clip_model.visual.transformer.resblocks[idx](x)
            x = self.forward_one_block(self.clip_model.visual.transformer.resblocks[-1], x)
            x = x[1:]
            x = x.permute(1, 0, 2)
            x = self.clip_model.visual.ln_post(x)
            x = x @ self.clip_model.visual.proj
            feat = x.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        feat = F.interpolate(feat, image_shape, mode = 'bilinear', align_corners = True)
        feat = F.normalize(feat, dim = 1)
        return feat.permute(0, 2, 3, 1)
    
    def run_mask_clip(self, rgb):
        with torch.no_grad():
            if self.device == 'cpu':
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device)
            else:
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device).half()
            features = self.extract_mask_clip_features(input, rgb.shape[-2:])[0].cpu()
    
        # cur_time = time.time()
        # image_vis = np.array(rgb.permute(1, 2, 0))
        # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("debug_chris/clean" + str(cur_time) + ".jpg", image_vis)
        return features
    
    def run_owl_sam_clip(self, rgb, valid_depth):
        with torch.no_grad():
            inputs = self.owl_processor(text=self.texts, images=rgb, return_tensors="pt")
            for input in inputs:
                inputs[input] = inputs[input].to(self.device)
            outputs = self.owl_model(**inputs)
            target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
            results = self.owl_processor.post_process_object_detection(outputs=outputs, threshold=0.15, target_sizes=target_sizes)
            if len(results[0]['boxes']) == 0:
                return None, None, None

            self.mask_predictor.set_image(rgb.permute(1,2,0).numpy())
            bounding_boxes = torch.stack(sorted(results[0]['boxes'], key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), revers = True), dim = 0)
            transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(bounding_boxes.detach().to(self.device), rgb.shape[-2:])
            masks, _, _= self.mask_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            masks = masks[:, 0, :, :].cpu()

            cur_time = time.time()
            image_vis = np.array(rgb.permute(1, 2, 0))
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite("debug_chris/clean" + str(cur_time) + ".jpg", image_vis)
            for idx, box in enumerate(bounding_boxes):
                tl_x, tl_y, br_x, br_y = box
                tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
                cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
            cv2.imwrite("debug_chris/seg" + str(cur_time) + ".jpg", image_vis)
    
            crops = []
            for box in bounding_boxes:
                tl_x, tl_y, br_x, br_y = box
                crops.append(self.clip_preprocess(transforms.ToPILImage()(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])))
            features = self.clip_model.encode_image(torch.stack(crops, dim = 0).to(self.device))
            features = F.normalize(features, dim = -1).cpu()

        feats = torch.empty((rgb.shape[-2], rgb.shape[-1], features.shape[-1]), dtype=features.dtype)
        final_mask = torch.full_like(valid_depth, fill_value=False, dtype=torch.bool)

        for (sam_mask, feature) in zip(masks.cpu(), features.cpu()):
            valid_mask = torch.logical_and(valid_depth, sam_mask)
            # cv2.imwrite('debug1.jpg', sam_mask.int().numpy() * 255)
            # cv2.imwrite('debug2.jpg', valid_depth.int().numpy() * 255)
            feats[valid_mask] = feature
            final_mask[valid_mask] = True
        
        return feats, final_mask

    def add_obs(
        self,
        obs: Observations,
        camera_K: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """Unpack an observation and convert it properly, then add to memory. Pass all other inputs into the add() function as provided."""
        rgb = self.fix_type(obs.rgb)
        depth = self.fix_type(obs.depth)
        xyz = self.fix_type(obs.xyz)
        camera_pose = self.fix_type(obs.camera_pose)
        base_pose = torch.from_numpy(
            np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        ).float()
        K = self.fix_type(obs.camera_K) if camera_K is None else camera_K

        # # Allow task_observations to provide semantic sensor
        # def _pop_with_task_obs_default(k, default=None):
        #     res = kwargs.pop(k, task_obs.get(k, None))
        #     if res is not None:
        #         res = self.fix_type(res)
        #     return res

        self.add(
            camera_pose=camera_pose,
            xyz=xyz,
            rgb=rgb,
            depth=depth,
            base_pose=base_pose,
            camera_K=K,
            *args,
            **kwargs,
        )

    def add(
        self,
        camera_pose: Tensor,
        rgb: Tensor,
        xyz: Optional[Tensor] = None,
        camera_K: Optional[Tensor] = None,
        feats: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
        base_pose: Optional[Tensor] = None,
        instance_image: Optional[Tensor] = None,
        instance_classes: Optional[Tensor] = None,
        instance_scores: Optional[Tensor] = None,
        obs: Optional[Observations] = None,
        xyz_frame: str = "camera",
        **info,
    ):
        """Add this to our history of observations. Also update the current running map.

        Parameters:
            camera_pose(Tensor): [4,4] cam_to_world matrix
            rgb(Tensor): N x 3 color points
            camera_K(Tensor): [3,3] camera instrinsics matrix -- usually pinhole model
            xyz(Tensor): N x 3 point cloud points in camera coordinates
            feats(Tensor): N x D point cloud features; D == 3 for RGB is most common
            base_pose(Tensor): optional location of robot base
            instance_image(Tensor): [H,W] image of ints where values at a pixel correspond to instance_id
            instance_classes(Tensor): [K] tensor of ints where class = instance_classes[instance_id]
            instance_scores: [K] of detection confidence score = instance_scores[instance_id]
            # obs: observations
        """
        # TODO: we should remove the xyz/feats maybe? just use observations as input?
        # TODO: switch to using just Obs struct?
        # Shape checking
        assert rgb.ndim == 3 or rgb.ndim == 2, f"{rgb.ndim=}: must be 2 or 3"
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        if isinstance(camera_pose, np.ndarray):
            camera_pose = torch.from_numpy(camera_pose)
        if depth is not None:
            assert (
                rgb.shape[:-1] == depth.shape
            ), f"depth and rgb image sizes must match; got {rgb.shape=} {depth.shape=}"
        assert xyz is not None or (camera_K is not None and depth is not None)
        if xyz is not None:
            assert (
                xyz.shape[-1] == 3
            ), "xyz must have last dimension = 3 for x, y, z position of points"
            assert rgb.shape == xyz.shape, "rgb shape must match xyz"
            # Make sure shape is correct here for xyz and any passed-in features
            if feats is not None:
                assert (
                    feats.shape[-1] == self.feature_dim
                ), f"features must match voxel feature dimenstionality of {self.feature_dim}"
                assert (
                    xyz.shape[0] == feats.shape[0]
                ), "features must be available for each point"
            else:
                pass
            if isinstance(xyz, np.ndarray):
                xyz = torch.from_numpy(xyz)
        if depth is not None:
            assert depth.ndim == 2 or xyz_frame == "world"
        if camera_K is not None:
            assert camera_K.ndim == 2, "camera intrinsics K must be a 3x3 matrix"
        assert (
            camera_pose.ndim == 2
            and camera_pose.shape[0] == 4
            and camera_pose.shape[1] == 4
        ), "Camera pose must be a 4x4 matrix representing a pose in SE(3)"
        assert (
            xyz_frame in VALID_FRAMES
        ), f"frame {xyz_frame} was not valid; should one one of {VALID_FRAMES}"

        # Apply a median filter to remove bad depth values when mapping and exploring
        # This is not strictly necessary but the idea is to clean up bad pixels
        if depth is not None and self.use_median_filter:
            median_depth = torch.from_numpy(
                scipy.ndimage.median_filter(depth, size=self.median_filter_size)
            )
            median_filter_error = (depth - median_depth).abs()

        # Get full_world_xyz
        if xyz is not None:
            if xyz_frame == "camera":
                full_world_xyz = (
                    torch.cat([xyz, torch.ones_like(xyz[..., [0]])], dim=-1)
                    @ camera_pose.T
                )[..., :3]
            elif xyz_frame == "world":
                full_world_xyz = xyz
            else:
                raise NotImplementedError(f"Unknown xyz_frame {xyz_frame}")
            # trimesh.transform_points(xyz, camera_pose)
        else:
            full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
                depth=depth.unsqueeze(0).unsqueeze(1),
                pose=camera_pose.unsqueeze(0),
                inv_intrinsics=torch.linalg.inv(camera_K[:3, :3]).unsqueeze(0),
            )
        # if feats is None:
        #     if not self.owl:
        #         feats = self.run_mask_clip(rgb.permute(2, 0, 1))
        #     else:
        #         feats, valid_depth = self.run_owl_sam_clip(rgb.permute(2, 0, 1), valid_depth)
        # add observations before we start changing things
        self.observations.append(
            Frame(
                camera_pose,
                camera_K,
                xyz,
                rgb,
                feats,
                depth,
                base_pose,
                info,
                full_world_xyz,
                xyz_frame=xyz_frame,
            )
        )

        valid_depth = torch.full_like(rgb[:, 0], fill_value=True, dtype=torch.bool)
        if depth is not None:
            valid_depth = (depth > self.min_depth) & (depth < self.max_depth)

            if self.use_derivative_filter:
                edges = get_edges(depth, threshold=self.derivative_filter_threshold)
                valid_depth = valid_depth & ~edges

            if self.use_median_filter:
                valid_depth = (
                    valid_depth
                    & (median_filter_error < self.median_filter_max_error).bool()
                )

        # Add to voxel grid
        if feats is not None:
            feats = feats[valid_depth].reshape(-1, feats.shape[-1])
        rgb = rgb[valid_depth].reshape(-1, 3)
        world_xyz = full_world_xyz.view(-1, 3)[valid_depth.flatten()]

        # TODO: weights could also be confidence, inv distance from camera, etc
        if world_xyz.nelement() > 0:
            self.voxel_pcd.add(world_xyz, features=feats, rgb=rgb, weights=None)

        if self._add_local_radius_points:
            # TODO: just get this from camera_pose?
            self._update_visited(camera_pose[:3, 3].to(self.map_2d_device))
        if base_pose is not None:
            self._update_visited(base_pose.to(self.map_2d_device))

        # Increment sequence counter
        self._seq += 1

    def _update_visited(self, base_pose: Tensor):
        """Update 2d map of where robot has visited"""
        # Add exploration here
        # Base pose can be whatever, going to assume xyt for now
        map_xy = ((base_pose[:2] / self.grid_resolution) + self.grid_origin[:2]).int()
        x0 = int(map_xy[0] - self._disk_size)
        x1 = int(map_xy[0] + self._disk_size + 1)
        y0 = int(map_xy[1] - self._disk_size)
        y1 = int(map_xy[1] + self._disk_size + 1)
        assert x0 >= 0
        assert y0 >= 0
        self._visited[x0:x1, y0:y1] += self._visited_disk

    def write_to_pickle(self, filename: str):
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts."""
        data = {}
        data["camera_poses"] = []
        data["camera_K"] = []
        data["base_poses"] = []
        data["xyz"] = []
        data["world_xyz"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        for frame in self.observations:
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(frame.camera_pose)
            data["base_poses"].append(frame.base_pose)
            data["camera_K"].append(frame.camera_K)
            data["xyz"].append(frame.xyz)
            data["world_xyz"].append(frame.full_world_xyz)
            data["rgb"].append(frame.rgb)
            data["depth"].append(frame.depth)
            data["feats"].append(frame.feats)
            for k, v in frame.info.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        (
            data["combined_xyz"],
            data["combined_feats"],
            data["combined_weights"],
            data["combined_rgb"],
        ) = self.voxel_pcd.get_pointcloud()
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def write_to_pickle_add_data(self, filename: str, newdata: dict):
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts."""
        data = {}
        data["camera_poses"] = []
        data["base_poses"] = []
        data["xyz"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        for key, value in newdata.items():
            data[key] = value
        for frame in self.observations:
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(frame.camera_pose)
            data["base_poses"].append(frame.base_pose)
            data["xyz"].append(frame.xyz)
            data["rgb"].append(frame.rgb)
            data["depth"].append(frame.depth)
            data["feats"].append(frame.feats)
            for k, v in frame.info.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        (
            data["combined_xyz"],
            data["combined_feats"],
            data["combined_weights"],
            data["combined_rgb"],
        ) = self.voxel_pcd.get_pointcloud()
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def fix_data_type(self, tensor) -> torch.Tensor:
        """make sure tensors are in the right format for this model"""
        # If its empty just hope we're handling that somewhere else
        if tensor is None:
            return None
        # Conversions
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        # Data types
        if isinstance(tensor, torch.Tensor):
            return tensor.float()
        else:
            raise NotImplementedError("unsupported data type for tensor:", tensor)

    def read_from_pickle(self, filename: str, num_frames: int = -1):
        """Read from a pickle file as above. Will clear all currently stored data first."""
        self.reset_cache()
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.exists(), f"No file found at {filename}"
        with filename.open("rb") as f:
            data = pickle.load(f)
        for i, (
            camera_pose,
            xyz,
            rgb,
            feats,
            depth,
            base_pose,
            K,
            world_xyz,
        ) in enumerate(
            zip(
                data["camera_poses"],
                data["xyz"],
                data["rgb"],
                data["feats"],
                data["depth"],
                data["base_poses"],
                data["camera_K"],
                data["world_xyz"],
            )
        ):
            # Handle the case where we dont actually want to load everything
            if num_frames > 0 and i >= num_frames:
                break

            camera_pose = self.fix_data_type(camera_pose)
            xyz = self.fix_data_type(xyz)
            rgb = self.fix_data_type(rgb)
            depth = self.fix_data_type(depth)
            if feats is not None:
                feats = self.fix_data_type(feats)
            base_pose = self.fix_data_type(base_pose)
            self.add(
                camera_pose=camera_pose,
                xyz=xyz,
                rgb=rgb,
                feats=feats,
                depth=depth,
                base_pose=base_pose,
                camera_K=K,
            )

    def recompute_map(self):
        """Recompute the entire map from scratch instead of doing incremental updates.
        This is a helper function which recomputes everything from the beginning.

        Currently this will be slightly inefficient since it recreates all the objects incrementally.
        """
        old_observations = self.observations
        self.reset_cache()
        for frame in old_observations:
            self.add(
                frame.camera_pose,
                frame.xyz,
                frame.rgb,
                frame.feats,
                frame.depth,
                frame.base_pose,
                frame.obs,
                **frame.info,
            )

    def get_2d_map(
        self, debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get 2d map with explored area and frontiers."""

        # Is this already cached? If so we don't need to go to all this work
        if self._map2d is not None and self._seq == self._2d_last_updated:
            return self._map2d

        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        xyz, _, counts, _ = self.voxel_pcd.get_pointcloud()
        if xyz is None:
            xyz = torch.zeros((0, 3))
            counts = torch.zeros((0))

        device = xyz.device
        xyz = ((xyz / self.grid_resolution) + self.grid_origin).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        # from home_robot.utils.point_cloud import show_point_cloud
        # show_point_cloud(xyz, rgb, orig=np.zeros(3))
        # xyz[xyz[:, -1] < 0, -1] = 0
        # show_point_cloud(xyz, rgb, orig=np.zeros(3))

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        grid_size = self.grid_size + [max_height]
        voxels = torch.zeros(grid_size, device=device)

        # Mask out obstacles only above a certain height
        obs_mask = xyz[:, -1] < max_height
        xyz = xyz[obs_mask, :]
        counts = counts[obs_mask][:, None]

        # voxels[x_coords, y_coords, z_coords] = 1
        voxels = scatter3d(xyz, counts, grid_size)

        # Compute the obstacle voxel grid based on what we've seen
        obstacle_voxels = voxels[:, :, min_height:]
        obstacles_soft = torch.sum(obstacle_voxels, dim=-1)
        obstacles = obstacles_soft > self.obs_min_density

        if self._remove_visited_from_obstacles:
            # Remove "visited" points containing observations of the robot
            obstacles *= (1 - self._visited).bool()

        if self.dilate_obstacles_kernel is not None:
            obstacles = binary_dilation(
                obstacles.float().unsqueeze(0).unsqueeze(0),
                self.dilate_obstacles_kernel,
            )[0, 0].bool()

        # Explored area = only floor mass
        # floor_voxels = voxels[:, :, :min_height]
        explored_soft = torch.sum(voxels, dim=-1)

        # Add explored radius around the robot, up to min depth
        # TODO: make sure lidar is supported here as well; if we do not have lidar assume a certain radius is explored
        explored_soft += self._visited
        explored = explored_soft > 0

        # Also shrink the explored area to build more confidence
        # That we will not collide with anything while moving around
        # if self.dilate_obstacles_kernel is not None:
        #    explored = binary_erosion(
        #        explored.float().unsqueeze(0).unsqueeze(0),
        #        self.dilate_obstacles_kernel,
        #    )[0, 0].bool()

        if self.smooth_kernel_size > 0:
            # Opening and closing operations here on explore
            explored = binary_erosion(
                binary_dilation(
                    explored.float().unsqueeze(0).unsqueeze(0), self.smooth_kernel
                ),
                self.smooth_kernel,
            )  # [0, 0].bool()
            explored = binary_dilation(
                binary_erosion(explored, self.smooth_kernel),
                self.smooth_kernel,
            )[0, 0].bool()

            # Obstacles just get dilated and eroded
            obstacles = binary_erosion(
                binary_dilation(
                    obstacles.float().unsqueeze(0).unsqueeze(0), self.smooth_kernel
                ),
                self.smooth_kernel,
            )[0, 0].bool()

        debug = True
        if debug:
            import matplotlib.pyplot as plt

            # TODO: uncomment to show the original world representation
            # from home_robot.utils.point_cloud import show_point_cloud
            # show_point_cloud(xyz, rgb / 255., orig=np.zeros(3))
            # TODO: uncomment to show voxel point cloud
            # from home_robot.utils.point_cloud import show_point_cloud
            # show_point_cloud(xyz, rgb/255., orig=self.grid_origin)

            plt.subplot(2, 2, 1)
            plt.imshow(obstacles_soft.detach().cpu().numpy())
            plt.title("obstacles soft")
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(explored_soft.detach().cpu().numpy())
            plt.title("explored soft")
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(obstacles.detach().cpu().numpy())
            plt.title("obstacles")
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(explored.detach().cpu().numpy())
            plt.axis("off")
            plt.title("explored")
            # plt.show()

        # Update cache
        self._map2d = (obstacles, explored)
        self._2d_last_updated = self._seq
        return obstacles, explored

    def xy_to_grid_coords(self, xy: torch.Tensor) -> Optional[np.ndarray]:
        """convert xy point to grid coords"""
        assert xy.shape[-1] == 2, "coords must be Nx2 or 2d array"
        # Handle convertion
        if isinstance(xy, np.ndarray):
            xy = torch.from_numpy(xy).float()
        grid_xy = (xy / self.grid_resolution) + self.grid_origin[:2]
        if torch.any(grid_xy >= self._grid_size_t) or torch.any(
            grid_xy < torch.zeros(2)
        ):
            return None
        else:
            return grid_xy

    def plan_to_grid_coords(
        self, plan_result: PlanResult
    ) -> Optional[List[torch.Tensor]]:
        """Convert a plan properly into grid coordinates"""
        if not plan_result.success:
            return None
        else:
            traj = []
            for node in plan_result.trajectory:
                traj.append(self.xy_to_grid_coords(node.state[:2]))
            return traj

    def grid_coords_to_xy(self, grid_coords: torch.Tensor) -> np.ndarray:
        """convert grid coordinate point to metric world xy point"""
        assert grid_coords.shape[-1] == 2, "grid coords must be an Nx2 or 2d array"
        return (grid_coords - self.grid_origin[:2]) * self.grid_resolution

    def grid_coords_to_xyt(self, grid_coords: np.ndarray) -> np.ndarray:
        """convert grid coordinate point to metric world xyt point"""
        res = torch.zeros(3)
        res[:2] = self.grid_coords_to_xy(grid_coords)
        return res

    def show(
        self, backend: str = "open3d", **backend_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""
        if backend == "open3d":
            return self._show_open3d(**backend_kwargs)
        elif backend == "pytorch3d":
            return self._show_pytorch3d(**backend_kwargs)
        else:
            raise NotImplementedError(
                f"Uknown backend {backend}, must be 'open3d' or 'pytorch3d"
            )

    def get_xyz_rgb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return xyz and rgb of the current map"""
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        return points, rgb

    def _show_pytorch3d(
        self, mock_plot: bool = False, **plot_scene_kwargs
    ):
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene

        from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
        from home_robot.utils.plotly_vis.camera import (
            add_camera_poses,
            colormap_to_rgb_strings,
        )

        points, rgb = self.get_xyz_rgb()

        # TODO: Xiaohan--need to normalize for older versions of pytorch3d. remove before merge
        rgb = rgb / 128.0

        traces = {}

        # Show points
        ptc = None
        if points is None and mock_plot:
            ptc = Pointclouds(
                points=[torch.zeros((2, 3))], features=[torch.zeros((2, 3))]
            )
        elif points is not None:
            ptc = Pointclouds(points=[points], features=[rgb])
        if ptc is not None:
            traces["Points"] = ptc

        # Show cameras
        # "Fused boxes": global_boxes,
        # "cameras": cameras,

        _default_plot_args = dict(
            xaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            yaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=800_000,
            boxes_plot_together=True,
            boxes_wireframe_width=3,
            aspectmode="cube",
        )
        fig = plot_scene_with_bboxes(
            plots={"Global scene": traces},
            **update(_default_plot_args, plot_scene_kwargs),
        )
        # Show cameras
        poses = [obs.camera_pose for obs in self.observations]
        add_camera_poses(fig, poses)
        return fig

    def sample_explored(self) -> Optional[np.ndarray]:
        """Return obstacle-free xy point in explored space"""
        obstacles, explored = self.get_2d_map()
        return self.sample_from_mask(~obstacles & explored)

    def sample_from_mask(self, mask: torch.Tensor) -> Optional[np.ndarray]:
        """Sample from any mask"""
        valid_indices = torch.nonzero(mask, as_tuple=False)
        if valid_indices.size(0) > 0:
            random_index = torch.randint(valid_indices.size(0), (1,))
            return self.grid_coords_to_xy(valid_indices[random_index])
        else:
            return None

    def xyt_is_safe(self, xyt: np.ndarray, robot: Optional[RobotModel] = None) -> bool:
        """Check to see if a given xyt position is known to be safe."""
        if robot is not None:
            raise NotImplementedError(
                "not currently checking against robot base geometry"
            )
        obstacles, explored = self.get_2d_map()
        # Convert xy to grid coords
        grid_xy = self.xy_to_grid_coords(xyt[:2])
        # Check to see if grid coords are explored and obstacle free
        if grid_xy is None:
            # Conversion failed - probably out of bounds
            return False
        obstacles, explored = self.get_2d_map()
        # Convert xy to grid coords
        grid_xy = self.xy_to_grid_coords(xyt[:2])
        # Check to see if grid coords are explored and obstacle free
        if grid_xy is None:
            # Conversion failed - probably out of bounds
            return False
        if robot is not None:
            # TODO: check against robot geometry
            raise NotImplementedError(
                "not currently checking against robot base geometry"
            )
        return True

    def _get_boxes_from_points(
        self,
        traversible: torch.Tensor,
        color: List[float],
        is_map: bool = True,
        height: float = 0.0,
        offset: Optional[np.ndarray] = None,
    ):
        """Get colored boxes for a mask"""
        # Get indices for all traversible locations
        traversible_indices = np.argwhere(traversible)
        # Traversible indices will be a 2xN array, so we need to transpose it.
        # Set to floor/max obs height and bright red
        if is_map:
            traversible_pts = self.grid_coords_to_xy(traversible_indices.T)
        else:
            traversible_pts = (
                traversible_indices.T - np.ceil([d / 2 for d in traversible.shape])
            ) * self.grid_resolution
        if offset is not None:
            traversible_pts += offset

        geoms = []
        for i in range(traversible_pts.shape[0]):
            center = np.array(
                [
                    traversible_pts[i, 0],
                    traversible_pts[i, 1],
                    self.obs_min_height + height,
                ]
            )
            dimensions = np.array(
                [self.grid_resolution, self.grid_resolution, self.grid_resolution]
            )

            # Create a custom geometry with red color
            mesh_box = open3d.geometry.TriangleMesh.create_box(
                width=dimensions[0], height=dimensions[1], depth=dimensions[2]
            )
            mesh_box.paint_uniform_color(color)  # Set color to red
            mesh_box.translate(center)

            # Visualize the red box
            geoms.append(mesh_box)
        return geoms

    def _get_open3d_geometries(
        self,
        orig: Optional[np.ndarray] = None,
        norm: float = 255.0,
        xyt: Optional[np.ndarray] = None,
        footprint: Optional[Footprint] = None,
        **backend_kwargs,
    ):
        """Show and return bounding box information and rgb color information from an explored point cloud. Uses open3d."""

        # Create a combined point cloud
        # pc_xyz, pc_rgb, pc_feats = self.get_data()
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        pcd = numpy_to_pcd(
            points.detach().cpu().numpy(), (rgb / norm).detach().cpu().numpy()
        )
        if orig is None:
            orig = np.zeros(3)
        geoms = create_visualization_geometries(pcd=pcd, orig=orig)

        # Get the explored/traversible area
        obstacles, explored = self.get_2d_map()
        traversible = explored & ~obstacles

        geoms += self._get_boxes_from_points(traversible, [0, 1, 0])
        geoms += self._get_boxes_from_points(obstacles, [1, 0, 0])

        if xyt is not None and footprint is not None:
            geoms += self._get_boxes_from_points(
                footprint.get_rotated_mask(self.grid_resolution, float(xyt[2])),
                [0, 0, 1],
                is_map=False,
                height=0.1,
                offset=xyt[:2],
            )

        return geoms

    def _get_o3d_robot_footprint_geometry(
        self,
        xyt: np.ndarray,
        dimensions: Optional[np.ndarray] = None,
        length_offset: float = 0,
    ):
        """Get a 3d mesh cube for the footprint of the robot. But this does not work very well for whatever reason."""
        # Define the dimensions of the cube
        if dimensions is None:
            dimensions = np.array(
                [0.2, 0.2, 0.2]
            )  # Cube dimensions (length, width, height)

        x, y, theta = xyt
        # theta = theta - np.pi/2

        # Create a custom mesh cube with the specified dimensions
        mesh_cube = open3d.geometry.TriangleMesh.create_box(
            width=dimensions[0], height=dimensions[1], depth=dimensions[2]
        )

        # Define the transformation matrix for position and orientation
        rotation_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        # Calculate the translation offset based on theta
        length_offset = 0
        x_offset = length_offset - (dimensions[0] / 2)
        y_offset = -1 * dimensions[1] / 2
        dx = (math.cos(theta) * x_offset) + (math.cos(theta - np.pi / 2) * y_offset)
        dy = (math.sin(theta + np.pi / 2) * y_offset) + (math.sin(theta) * x_offset)
        translation_vector = np.array(
            [x + dx, y + dy, 0]
        )  # Apply offset based on theta
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

        # Apply the transformation to the cube
        mesh_cube.transform(transformation_matrix)

        # Set the color of the cube to blue
        mesh_cube.paint_uniform_color([0.0, 0.0, 1.0])  # Set color to blue

        return mesh_cube

    def _show_open3d(
        self,
        orig: Optional[np.ndarray] = None,
        norm: float = 255.0,
        xyt: Optional[np.ndarray] = None,
        footprint: Optional[Footprint] = None,
        **backend_kwargs,
    ):
        """Show and return bounding box information and rgb color information from an explored point cloud. Uses open3d."""

        # get geometries so we can use them
        geoms = self._get_open3d_geometries(
            orig, norm, xyt=xyt, footprint=footprint
        )

        # Show the geometries of where we have explored
        open3d.visualization.draw_geometries(geoms)

        # Returns xyz and rgb for further inspection
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        return points, rgb