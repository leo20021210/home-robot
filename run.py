import rospy
from std_msgs.msg import Float64MultiArray
import sys
sys.path.append('/home/hello-robot/hello-robot-peiqi')
from util import navigate
import numpy as np

from home_robot_hw.remote import StretchClient
from util import navigate

robot = StretchClient()

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    paths = data.data
    i = 0
    while i < len(paths):
        x = -paths[i]
        y = paths[i + 1]
        navigate(robot, np.array([x, y, 0]))
        i += 2

def run():
    # Reset robot
    print("Resetting robot...")
    robot.reset()
    print(robot.nav.get_base_pose())

    #rospy.init_node('listener', anonymous=True)
    #pub = rospy.Publisher('pos', Float64MultiArray, queue_size = 1)
    rospy.Subscriber('path', Float64MultiArray, callback)

    #while not rospy.is_shutdown():
    #    robot.nav.get_base_pose()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    run()
