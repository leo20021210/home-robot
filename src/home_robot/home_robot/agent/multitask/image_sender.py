import zmq
from home_robot_hw.remote import StretchClient
import numpy as np

import threading
from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from rospy import Subscriber
from sensor_msgs.msg import CameraInfo, Image

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    A = np.array(A)
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

class ImageSender:
    def __init__(self, 
        stop_and_photo = False, 
        ip = '100.107.224.62', 
        image_port = 5555,
        text_port = 5556,
        color_name = "/camera/color",
        depth_name = "/camera/aligned_depth_to_color",
        camera_name = "/camera_pose",
        slop_time_seconds = 0.05,
        queue_size = 100,
    ):
        context = zmq.Context()
        self.img_socket = context.socket(zmq.REQ)
        self.img_socket.connect('tcp://' + str(ip) + ':' + str(image_port))
        self.text_socket = context.socket(zmq.REQ)
        self.text_socket.connect('tcp://' + str(ip) + ':' + str(text_port))

        # self.color_name = color_name
        # self.depth_name = depth_name
        # self.camera_name = camera_name
        # self.slop_time_seconds = slop_time_seconds
        # self.queue_size = queue_size
        # self._read_camera_intrinsics()
        # if not stop_and_photo:
        #     self._start_synchronizers()

        # self.lock = threading.Lock() 
        # self.thread = threading.Thread(target=self.run_task)
        # self.thread.daemon = True
        # self.thread.start()

    def query_text(self, text):
        self.text_socket.send_string(text)
        return recv_array(self.text_socket)
        
    def send_images(self, obs):
        rgb = obs.rgb
        depth = obs.depth
        camer_K = obs.camera_K
        camera_pose = obs.camera_pose
        data = np.concatenate((depth.shape, rgb.flatten(), depth.flatten(), camer_K.flatten(), camera_pose.flatten()))
        send_array(self.img_socket, data)
        print(self.img_socket.recv_string())

    # def _read_camera_intrinsics(self):
    #     self._color_sub, self._color_camera_info = self._start_camera(self.color_name)
    #     self._depth_sub, self._depth_camera_info = self._start_camera(self.depth_name)
    #     self.K = self._depth_camera_info.K
    #     self.K = np.array(self.K).reshape(3, 3)
    #     self.K[0, 0], self.K[1, 1] = self.K[1, 1], self.K[0, 0]
    #     self.K[0, 2], self.K[1, 2] = self.K[1, 2], self.K[0, 2]
    #     self.fx = self.K[0, 0]
    #     self.fy = self.K[1, 1]
    #     self.px = self.K[0, 2]
    #     self.py = self.K[1, 2]

    # def _start_synchronizers(self):
    #     self.rgb_image = None
    #     self.depth_image = None
    #     self.camera_pose = None
    #     self.height = self._depth_camera_info.width
    #     self.width = self._depth_camera_info.height
    #     self._camera_sub = Subscriber(self.camera_pose_topic, PoseStamped, self._pose_callback, queue_size = 1)

    # def _start_camera(self, name):
    #     '''
    #         Start a ros camera topic (e.g. depth, rgb)
    #         return:
    #             - a subscriber listening to that camera topic
    #             - camera info, e.g. camera K
    #     '''
    #     camera_info_topic = name + "/camera_info"
    #     if self.verbose:
    #         print("Waiting for camera info on", self._camera_info_topic + "...")
    #     cam_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    #     topic = name + "/image_raw"
    #     return Subscriber(topic, Image), cam_info

robot = StretchClient()
ImageSender().send_images(robot.get_observation())