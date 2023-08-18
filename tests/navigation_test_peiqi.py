import numpy as np

from home_robot_hw.remote import StretchClient

# Loose tolerances just to test that the robot moved reasonably
POS_TOL = 0.1
YAW_TOL = 0.2


if __name__ == "__main__":
    print("Initializing...")
    robot = StretchClient()

    # Reset robot
    print("Resetting robot...")
    robot.reset()
    print(robot.nav.get_base_pose())

    # Head movement
    print("Testing robot head movement...")
    robot.head.look_at_ee()
    robot.head.look_front()

    print("Confirm that the robot head has moved accordingly.")
    input("(press enter to continue)")

    # Navigation
    print("Testing robot navigation...")
    robot.switch_to_navigation_mode()

    print("The robot locates at " + str(robot.nav.get_base_pose()))
    xyt_goal = [1, 4.5, 0.5]
    print("The robot navigates to " + str(xyt_goal))
    i = 0
    while True:
        i += 1
        print("Move " + str(i))
        robot.nav.navigate_to(xyt_goal)
        xyt_curr = robot.nav.get_base_pose()
        print("The robot currently loactes at " + str(xyt_curr))
        if np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL) and np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL):
            print(str(i) + " moves takes our robot to " + str(xyt_goal))
            break

    print(f"Confirm that the robot moved to {xyt_goal} (forward left, facing right)")
    input("(press enter to continue)")


    print("The robot locates at " + str(robot.nav.get_base_pose()))
    xyt_goal = [-4, 4.5, np.pi / 2]
    print("The robot navigates to " + str(xyt_goal))
    i = 0
    while True:
        i += 1
        print("Move " + str(i))
        robot.nav.navigate_to(xyt_goal)
        xyt_curr = robot.nav.get_base_pose()
        print("The robot currently loactes at " + str(xyt_curr))
        if np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL) and np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL):
            print(str(i) + " moves takes our robot to " + str(xyt_goal))
            break

    print("The robot locates at " + str(robot.nav.get_base_pose()))
    xyt_goal = [-7, 6, 0]
    print("The robot navigates to " + str(xyt_goal))
    i = 0
    while True:
        i += 1
        print("Move " + str(i))
        robot.nav.navigate_to(xyt_goal)
        xyt_curr = robot.nav.get_base_pose()
        print("The robot currently loactes at " + str(xyt_curr))
        if np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL) and np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL):
            print(str(i) + " moves takes our robot to " + str(xyt_goal))
            break

    print(f"Confirm that the robot moved to {xyt_goal} (forward left, facing right)")
    input("(press enter to continue)")

    print("Test complete!")
