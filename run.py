import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np

from home_robot_hw.remote import StretchClient
import zmq

robot = StretchClient()

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://172.24.71.253:5555")

def navigate(robot, goal):
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
    robot.reset()
    print(robot.nav.get_base_pose())
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.24.71.253:5555")
    while True:
        start_xy = robot.nav.get_base_pose()
        end_x = float(input("Enter x:"))
        print("end_x =", end_x)
        end_y = float(input("Enter y:"))
        print("end_y =", end_y)
        send_array(socket, start_xy)
        print(socket.recv_string())
        send_array(socket, [end_x, end_y])
        print(socket.recv_string())
        socket.send_string("Waiting for path")
        paths = recv_array(socket)
        for path in paths:
            #print(path)
            navigate(robot, path)

if __name__ == '__main__':
    run()
