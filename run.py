import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np

from home_robot_hw.remote import StretchClient
#import stretch_body.robot
import zmq
import time

robot = StretchClient()
#robot.switch_to_navigation_mode()
#robot.move_to_nav_posture()
#robot = stretch_body.robot.Robot()
#robot.startup()

POS_TOL = 0.1
YAW_TOL = 0.2

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://172.24.71.253:5555")

def navigate(robot, xyt_goal):
    while True:
        robot.nav.navigate_to(xyt_goal)
        xyt_curr = robot.nav.get_base_pose()
        print("The robot currently loactes at " + str(xyt_curr))
        time.sleep(0.5)
        if np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL) and np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL):
            print("The robot is finally at " + str(xyt_goal))
            break

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    paths = data.data
    i = 0
    while i < len(paths):
        x = -paths[i]
        y = paths[i + 1]
        navigate(robot, np.array([x, y, 0]))
        i += 2

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    A = np.array(A)
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

# use zmq to receive a numpy array
def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def run():
    # Reset robot
    print("Resetting robot...")
    #robot.reset()
    print(robot.nav.get_base_pose())
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.24.71.253:5555")
    mode = int(input("test mode (1 for debugging, 2 for open vocab): "))
    while True:
        start_xy = robot.nav.get_base_pose()
        #start_xy = [robot.base.status['x'], robot.base.status['y'], robot.base.status['theta']]
        if mode == 1:
            end_x = float(input("Enter x:"))
            print("end_x = ", end_x)
            end_y = float(input("Enter y:"))
            print("end_y = ", end_y)
        else:
            A = str(input("Enter A: "))
            print("A = ", A)
            B = str(input("Enter B: "))
            print("B = ", B)
        send_array(socket, start_xy)
        print(socket.recv_string())
        if mode == 1:
            send_array(socket, [end_x, end_y])
            print(socket.recv_string())
        else:
            socket.send_string(A)
            print(socket.recv_string())
            socket.send_string(B)
            print(socket.recv_string())
        socket.send_string("Waiting for path")
        paths = recv_array(socket)
        print(paths)
        for path in paths:
            navigate(robot, path)
        #xyt = robot.nav.get_base_pose()
        #xyt[2] = xyt[2] + np.pi / 2
        #navigate(robot, xyt)
            #cur_xy = [robot.base.status['x'], robot.base.status['y'], robot.base.status['theta']]
            #dis = np.linalg.norm(path[:2] - cur_xy[:2])
            #rotate = path[2] - cur_xy[2]
            #print(cur_xy)
            #print(path)
            #print(dis, rotate)
            #robot.base.rotate_by(rotate)
            #robot.push_command()
            #robot.base.left_wheel.wait_until_at_setpoint()
            #print("Rotated")
            #robot.base.translate_by(dis)
            #robot.push_command()
            #robot.base.left_wheel.wait_until_at_setpoint()
            #print("Translated")

if __name__ == '__main__':
    run()
