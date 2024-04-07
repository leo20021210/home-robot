# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import threading
from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
# from message_filters import ApproximateTimeSynchronizer, Subscriber
from rospy import Subscriber
from sensor_msgs.msg import CameraInfo, Image

from home_robot.utils.image import Camera
from home_robot_hw.ros.msg_numpy import image_to_numpy

from home_robot_hw.ros.utils import matrix_from_pose_msg

from home_robot.utils.voxel import VoxelizedPointcloud
from scannet import CLASS_LABELS_200

import cv2
import sophus as sp

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, OwlViTForObjectDetection
import clip
from torchvision import transforms

import os
import wget

from matplotlib import pyplot as plt

DEFAULT_COLOR_TOPIC = "/camera/color"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color"
DEFAULT_CAMERA_TOPIC = "/camera_pose"
# DEFAULT_POSE_TOPIC = "/state_estimator/pose_filtered"


class SynchronizedSensors(object):
    """Quick class to use a time synchronizer to collect sensor data to speed up the robot execution."""
    def _start_camera(self, name):
        camera_info_topic = name + "/camera_info"
        if self.verbose:
            print("Waiting for camera info on", camera_info_topic + "...")
        cam_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        topic = name + "/image_raw"
        # return Subscriber(topic, Image), cam_info
        if 'depth' in name:
            print('Starting depth subscriber')
            return Subscriber(topic, Image, self._depth_callback, queue_size = 1), cam_info
        else:
            print('Starting RGB subscriber')
            return Subscriber(topic, Image, self._rgb_callback, queue_size = 1), cam_info


    def depth_to_xyz(self, depth):
        """get depth from numpy using simple pinhole camera model"""
        indices = np.indices((self.height, self.width), dtype=np.float32).transpose(
            1, 2, 0
        )
        z = depth
        # pixel indices start at top-left corner. for these equations, it starts at bottom-left
        x = (indices[:, :, 1] - self.px) * (z / self.fx)
        y = (indices[:, :, 0] - self.py) * (z / self.fy)
        # Should now be height x width x 3, after this:
        xyz = np.stack([x, y, z], axis=-1)
        return xyz
    
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
    
    def run_mask_clip(self, rgb, mask, world_xyz):
        if len(rgb.shape) > 3:
            rgb = rgb[0]
        with torch.no_grad():
            if self.device == 'cpu':
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device)
            else:
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device).half()
            features = self.extract_mask_clip_features(input, rgb.shape[-2:])[0].cpu()
    
        valid_xyz = world_xyz[~mask[0]]
        features = features[~mask[0, 0]]
        valid_rgb = rgb.permute(1, 2, 0)[~mask[0, 0]]
        if len(valid_xyz) == 0:
            return None, None, None
        else:
            return valid_xyz, features, valid_rgb
    
    def run_owl_sam_clip(self, rgb, mask, world_xyz):
        if len(rgb.shape) > 3:
            rgb = rgb[0]
        if len(mask.shape) > 3:
            mask = mask[0]
        if len(world_xyz.shape) > 3:
            world_xyz = world_xyz[0]
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
            transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(results[0]['boxes'].detach().to(self.device), rgb.shape[-2:])
            masks, _, _= self.mask_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            masks = masks[:, 0, :, :].cpu()

            # if True:
            #     image_vis = np.array(rgb.permute(1, 2, 0))
            #     image = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite("Clean.jpg", image)
            #     for box in results[0]['boxes']:
            #         tl_x, tl_y, br_x, br_y = box
            #         tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
            #         image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
            #         cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
            #         segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
            #         # for vis_mask in masks: 
            #         #     segmentation_color_map[mask.detach().cpu().numpy()] = [0, 255, 0]
            #         # image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
            #         cv2.imwrite("seg.jpg", image_vis)
    
            crops = []
            for box in results[0]['boxes']:
                tl_x, tl_y, br_x, br_y = box
                crops.append(self.clip_preprocess(transforms.ToPILImage()(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])))
            features = self.clip_model.encode_image(torch.stack(crops, dim = 0).to(self.device))
            features = F.normalize(features, dim = -1).cpu()

        for (sam_mask, feature) in zip(masks.cpu(), features.cpu()):
            valid_mask = torch.logical_and(~mask[0], sam_mask)
            # plt.imsave('a.jpg', ~mask[0])
            # plt.imsave('b.jpg', sam_mask)
            # plt.imsave('c.jpg', valid_mask)
            valid_xyz = world_xyz[valid_mask]
            if valid_xyz.shape[0] == 0:
                return None, None, None
            feature = feature.repeat(valid_xyz.shape[0], 1)
            valid_rgb = rgb.permute(1, 2, 0)[valid_mask]
        
        return valid_xyz, feature, valid_rgb

    def add_image_to_voxel_map(self, rgb, depth, world_xyz):
        while len(rgb.shape) < 4:
            rgb = rgb.unsqueeze(0)
        while len(depth.shape) < 4:
            depth = depth.unsqueeze(0)
        while len(world_xyz.shape) < 4:
            world_xyz = world_xyz.unsqueeze(0)
        mask = torch.logical_or(depth > 3, depth < 0.1)
        # print(rgb.shape, world_xyz.shape, depth.shape)
        if self.owl:
            valid_xyz, feature, valid_rgb = self.run_owl_sam_clip(rgb, mask, world_xyz)
        else:
            valid_xyz, feature, valid_rgb = self.run_mask_clip(rgb, mask, world_xyz)
        # print(valid_xyz)
        if valid_xyz is not None:
            # print(valid_xyz.shape, feature.shape, valid_rgb.shape)
            self.voxel_pcd.add(points = valid_xyz, 
              features = feature,
              rgb = valid_rgb,)

    def _rgb_callback(self, color):
        time_step = color.header.stamp.to_sec() - self._t
        time_step = (time_step // self.slop_time_seconds) % self.queue_size
        if time_step in self.depth_images and time_step in self.camera_poses:
            camera_time, camera_pose = self.camera_poses[time_step]
            depth_time, depth_image = self.depth_images[time_step]
            if abs(depth_time - color.header.stamp.to_sec()) < self.slop_time_seconds and abs(camera_time - color.header.stamp.to_sec()) < self.slop_time_seconds:
                print('rgb comes from', color.header.stamp.to_sec() - self._t)
                print('depth comes from', depth_time - self._t)
                print('pose comes from', camera_time - self._t)
                xyz = self.depth_to_xyz(depth_image)
                world_xyz = (
                    np.concatenate((xyz, np.ones_like(xyz[..., [0]])), axis=-1)
                    @ camera_pose.T
                )[..., :3]
                world_xyz = torch.from_numpy(np.array(world_xyz))
                rgb_image = image_to_numpy(color)
                rgb_image = np.rot90(rgb_image, k = 3)
                rgb_image = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1)
                depth_image = torch.from_numpy(np.array(depth_image))
                self.add_image_to_voxel_map(rgb_image, depth_image, world_xyz)

    def _depth_callback(self, depth):
        # self._times['depth'] = depth.header.stamp.to_sec()
        depth_image = image_to_numpy(depth) / 1000.0
        depth_image = np.rot90(depth_image, k = 3)
        # self.depth_image = torch.from_numpy(np.array(depth_image))
        time_step = depth.header.stamp.to_sec() - self._t
        time_step = (time_step // self.slop_time_seconds) % self.queue_size
        self.depth_images[time_step] = (depth.header.stamp.to_sec(), depth_image)

    def _pose_callback(self, pose):
        # self._times['camera'] = pose.header.stamp.to_sec()
        time_step = pose.header.stamp.to_sec() - self._t
        time_step = (time_step // self.slop_time_seconds) % self.queue_size
        self.camera_poses[time_step] = (pose.header.stamp.to_sec(), np.array(matrix_from_pose_msg(pose.pose)))
        # if self.depth_image is not None:
        #     xyz = self.depth_to_xyz(self.depth_image.numpy())
        #     camera_pose = np.array(matrix_from_pose_msg(pose.pose))
        #     world_xyz = (
        #         np.concatenate((xyz, np.ones_like(xyz[..., [0]])), axis=-1)
        #         @ camera_pose.T
        #     )[..., :3]
        #     self.world_xyz = torch.from_numpy(np.array(world_xyz))

    # def get_times(self) -> Dict[str, float]:
    #     """Get the times for all measurements"""
    #     return self._times

    def __init__(
        self,
        color_name,
        depth_name,
        camera_pose_topic,
        # pose_topic,
        verbose=True,
        slop_time_seconds=0.1,
        queue_size = 200,
        owl = False,
        device = 'cuda'
    ):
        self.slop_time_seconds = slop_time_seconds
        self.queue_size = queue_size
        self.verbose = verbose
        self._t = rospy.Time.now().to_sec()
        self.camera_poses = dict()
        self.depth_images = dict()
        self._lock = threading.Lock()

        self.owl = owl
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device)
        self.clip_model.eval()
        if owl:
            self.owl_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
            self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval().to(device)
            if not os.path.exists('sam_vit_b_01ec64.pth'):
                wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', out = 'sam_vit_b_01ec64.pth')
            sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
            self.mask_predictor = SamPredictor(sam)
            self.mask_predictor.model = self.mask_predictor.model.eval().to(device)
            self.texts = [['a photo of ' + text for text in CLASS_LABELS_200]]
        self.voxel_pcd = VoxelizedPointcloud()

        if verbose:
            print("Creating subs...")
        self._color_sub, self._color_camera_info = self._start_camera(color_name)
        self._depth_sub, self._depth_camera_info = self._start_camera(depth_name)
        self.K = self._depth_camera_info.K
        self.K = np.array(self.K).reshape(3, 3)
        self.K[0, 0], self.K[1, 1] = self.K[1, 1], self.K[0, 0]
        self.K[0, 2], self.K[1, 2] = self.K[1, 2], self.K[0, 2]
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.px = self.K[0, 2]
        self.py = self.K[1, 2]
        self.height = self._depth_camera_info.width
        self.width = self._depth_camera_info.height
        # print(self.fx, self.fy, self.px, self.py)
        # self._camera_sub = Subscriber(camera_pose_topic, PoseStamped)
        self._camera_sub = Subscriber(camera_pose_topic, PoseStamped, self._pose_callback, queue_size = 1)

        # Store time information
        # self._times = {}

        if verbose:
            print("Time synchronizer...")
        # self._sync = ApproximateTimeSynchronizer(
        #     # [self._color_sub, self._depth_sub, self._camera_sub, self._pose_sub],
        #     [self._color_sub, self._depth_sub, self._camera_sub],
        #     queue_size=50,
        #     slop=slop_time_seconds,
        # )
        # self._sync.registerCallback(self._callback)


if __name__ == "__main__":
    rospy.init_node("sync_sensors_test")
    sensor = SynchronizedSensors(
        color_name=DEFAULT_COLOR_TOPIC,
        depth_name=DEFAULT_DEPTH_TOPIC,
        camera_pose_topic=DEFAULT_CAMERA_TOPIC,
        # pose_topic=DEFAULT_POSE_TOPIC,
    )
    rate = rospy.Rate(1)
    t0 = rospy.Time.now()
    try:
        while not rospy.is_shutdown():
            t1 = rospy.Time.now()
            # print((t1 - t0).to_sec())
            # times = sensor.get_times()
            # for k, v in times.items():
            #     print("-", k, v - t0.to_sec())
            rate.sleep()
    finally:
        print('Stop streaming images and write memory data')
        torch.save(sensor.voxel_pcd, 'memory_mahi.pt')
