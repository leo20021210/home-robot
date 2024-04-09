# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import datetime
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from loguru import logger

from home_robot.agent.multitask import Parameters
from home_robot.core.robot import RobotClient
from home_robot.mapping.voxel import (
    SparseVoxelMapV2,
    SparseVoxelMapNavigationSpace,
    plan_to_frontier,
)
from home_robot.motion import (
    ConfigurationSpace,
    PlanResult,
    RRTConnect,
    Shortcut,
    SimplifyXYT,
)

import cv2


class RobotAgentV2:
    """Basic demo code. Collects everything that we need to make this work."""

    _retry_on_fail = False

    def __init__(
        self,
        robot: RobotClient,
        parameters: Dict[str, Any],
        voxel_map: Optional[SparseVoxelMapV2] = None,
    ):
        print('------------------------YOU ARE NOW RUNNING PEIQI CODES V2-----------------')
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot

        self.normalize_embeddings = True
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]
        self.obs_count = 0
        self.obs_history = []
        self.guarantee_instance_is_reachable = (
            parameters.guarantee_instance_is_reachable
        )

        # Expanding frontier - how close to frontier are we allowed to go?
        self.default_expand_frontier_size = parameters["default_expand_frontier_size"]

        if voxel_map is not None:
            self.voxel_map = voxel_map
        else:
            self.voxel_map = SparseVoxelMapV2(
                resolution=parameters["voxel_size"],
                local_radius=parameters["local_radius"],
                obs_min_height=parameters["obs_min_height"],
                obs_max_height=parameters["obs_max_height"],
                min_depth=parameters["min_depth"],
                max_depth=parameters["max_depth"],
                pad_obstacles=parameters["pad_obstacles"],
                add_local_radius_points=parameters.get(
                    "add_local_radius_points", default=True
                ),
                remove_visited_from_obstacles=parameters.get(
                    "remove_visited_from_obstacles", default=False
                ),
                obs_min_density=parameters["obs_min_density"],
                smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
                use_median_filter=parameters.get("filters/use_median_filter", False),
                median_filter_size=parameters.get("filters/median_filter_size", 5),
                median_filter_max_error=parameters.get(
                    "filters/median_filter_max_error", 0.01
                ),
                use_derivative_filter=parameters.get(
                    "filters/use_derivative_filter", False
                ),
                derivative_filter_threshold=parameters.get(
                    "filters/derivative_filter_threshold", 0.5
                )
            )

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot.get_robot_model(),
            step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters[
                "dilate_frontier_size"
            ],  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )

        # Create a simple motion planner
        self.planner = RRTConnect(self.space, self.space.is_valid)
        if parameters["motion_planner"]["shortcut_plans"]:
            self.planner = Shortcut(
                self.planner, parameters["motion_planner"]["shortcut_iter"]
            )
        if parameters["motion_planner"]["simplify_plans"]:
            self.planner = SimplifyXYT(
                self.planner, min_step=0.05, max_step=1.0, num_steps=8
            )

        timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    def get_navigation_space(self) -> ConfigurationSpace:
        """Returns reference to the navigation space."""
        return self.space

    def rotate_in_place(self, steps: int = 12, visualize: bool = True) -> bool:
        """Simple helper function to make the robot rotate in place. Do a 360 degree turn to get some observations (this helps debug the robot and create a nice map).

        Returns:
            executed(bool): false if we did not actually do any rotations"""
        logger.info("Rotate in place")
        self.robot.move_to_nav_posture()
        if steps <= 0:
            return False
        step_size = 2 * np.pi / steps
        i = 0
        while i < steps:
            self.robot.navigate_to([0, 0, step_size], relative=True, blocking=True)
            # TODO remove debug code
            # print(i, self.robot.get_base_pose())
            self.update()
            if self.robot.last_motion_failed():
                # We have a problem!
                self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                i = 0
            else:
                i += 1

            if visualize:
                self.voxel_map.show(
                    orig=np.zeros(3),
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                )

        return True

    def get_observations(self, task=None):
        from home_robot.utils.rpc import get_obj_centric_world_representation

        instances = self.voxel_map.get_instances()
        world_representation = get_obj_centric_world_representation(
            instances,
            self.parameters["vlm_context_length"],
            self.parameters["sample_strategy"],
            task=task,
        )
        return world_representation

    def say(self, msg: str):
        """Provide input either on the command line or via chat client"""
        print(msg)

    def ask(self, msg: str) -> str:
        """Receive input from the user either via the command line or something else"""
        return input(msg)

    def get_command(self):
        if (
            "command" in self.parameters.data.keys()
        ):  # TODO: this was breaking. Should this be a class method
            return self.parameters["command"]
        else:
            return self.ask("please type any task you want the robot to do: ")

    def __del__(self):
        """Clean up at the end if possible"""
        print("... Done.")
        

    def update(self, visualize_map=False):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        for tilt in [-0.3, -0.6]:
            for pan in [1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5]:
                self.robot.head.set_pan_tilt(pan = pan, tilt = tilt)
                time.sleep(0.2)
                cv2.imwrite('debug_chris/debug' + str(self.obs_count) + '.jpg', self.robot.head.get_images()[0][:, :, [2, 1, 0]])
                obs = self.robot.get_observation()
                self.obs_history.append(obs)
                self.obs_count += 1
        # obs_count = self.obs_count

        self.voxel_map.add_obs(obs)
        # Add observation - helper function will unpack it
        if visualize_map:
            # Now draw 2d maps to show waht was happening
            self.voxel_map.get_2d_map(debug=True)

    def go_home(self):
        """Simple helper function to send the robot home safely after a trial."""
        print("Go back to (0, 0, 0) to finish...")
        print("- change posture and switch to navigation mode")
        self.robot.move_to_nav_posture()
        # self.robot.head.look_close(blocking=False)
        self.robot.switch_to_navigation_mode()

        print("- try to motion plan there")
        start = self.robot.get_base_pose()
        goal = np.array([0, 0, 0])
        print(
            f"- Current pose is valid: {self.space.is_valid(self.robot.get_base_pose())}"
        )
        print(f"-   start pose is valid: {self.space.is_valid(start)}")
        print(f"-    goal pose is valid: {self.space.is_valid(goal)}")
        res = self.planner.plan(start, goal)
        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("- executing full plan to home!")
            self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            print("Done!")
        else:
            print("Can't go home; planning failed!")

    def run_exploration(
        self,
        rate: int = 10,
        manual_wait: bool = False,
        explore_iter: int = 3,
        try_to_plan_iter: int = 10,
        dry_run: bool = False,
        visualize: bool = False,
        task_goal: str = None,
        go_home_at_end: bool = False,
        go_to_start_pose: bool = True,
        show_goal: bool = False,
    ):
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        self.robot.move_to_nav_posture()

        if go_to_start_pose:
            print("Go to (0, 0, 0) to start with...")
            self.robot.navigate_to([0, 0, 0])

        all_starts = []
        all_goals = []

        # Explore some number of times
        no_success_explore = True
        for i in range(explore_iter):
            print("\n" * 2)
            print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start, verbose=True)
            # if start is not valid move backwards a bit
            if not start_is_valid:
                print("Start not valid. back up a bit.")

                # TODO: debug here -- why start is not valid?
                self.update()
                # self.save_svm("", filename=f"debug_svm_{i:03d}.pkl")
                print(f"robot base pose: {self.robot.get_base_pose()}")

                print("--- STARTS ---")
                for a_start, a_goal in zip(all_starts, all_goals):
                    print(
                        "start =",
                        a_start,
                        self.space.is_valid(a_start),
                        "goal =",
                        a_goal,
                        self.space.is_valid(a_goal),
                    )

                self.robot.navigate_to([-0.1, 0, 0], relative=True)
                continue

            print("       Start:", start)
            # sample a goal
            res = plan_to_frontier(
                start,
                self.planner,
                self.space,
                self.voxel_map,
                try_to_plan_iter=try_to_plan_iter,
                visualize=False,  # visualize,
                expand_frontier_size=self.default_expand_frontier_size,
            )

            # if it succeeds, execute a trajectory to this position
            if res.success:
                no_success_explore = False
                print("Plan successful!")
                for i, pt in enumerate(res.trajectory):
                    print(i, pt.state)
                all_starts.append(start)
                all_goals.append(res.trajectory[-1].state)
                if visualize:
                    print("Showing goal location:")
                    robot_center = np.zeros(3)
                    robot_center[:2] = self.robot.get_base_pose()[:2]
                    self.voxel_map.show(
                        orig=robot_center,
                        xyt=res.trajectory[-1].state,
                        footprint=self.robot.get_robot_model().get_footprint(),
                    )
                if not dry_run:
                    self.robot.execute_trajectory(
                        [pt.state for pt in res.trajectory],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                    )
            else:
                if self._retry_on_fail:
                    print("Failed. Try again!")
                    continue
                else:
                    print("Failed. Quitting!")
                    break

            if self.robot.last_motion_failed():
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("ROBOT IS STUCK! Move back!")

                # help with debug TODO: remove
                self.update()
                # self.save_svm(".")
                print(f"robot base pose: {self.robot.get_base_pose()}")

                r = np.random.randint(3)
                if r == 0:
                    self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                elif r == 1:
                    self.robot.navigate_to(
                        [0, 0, np.pi / 4], relative=True, blocking=True
                    )
                elif r == 2:
                    self.robot.navigate_to(
                        [0, 0, -np.pi / 4], relative=True, blocking=True
                    )

            # Append latest observations
            self.update()
            # self.save_svm("", filename=f"debug_svm_{i:03d}.pkl")
            if visualize:
                # After doing everything - show where we will move to
                robot_center = np.zeros(3)
                robot_center[:2] = self.robot.get_base_pose()[:2]
                self.voxel_map.show(
                    orig=robot_center,
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                )

            if manual_wait:
                input("... press enter ...")

        # if it fails to find any frontier in the given iteration, simply quit in sim
        if no_success_explore:
            print("The robot did not explore at all, force quit in sim")
            self.robot.force_quit = True

        if go_home_at_end:
            # Finally - plan back to (0,0,0)
            print("Go back to (0, 0, 0) to finish...")
            start = self.robot.get_base_pose()
            goal = np.array([0, 0, 0])
            res = self.planner.plan(start, goal)
            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Full plan to home:")
                for i, pt in enumerate(res.trajectory):
                    print("-", i, pt.state)
                if not dry_run:
                    self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            else:
                print("WARNING: planning to home failed!")
