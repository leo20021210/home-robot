import numpy as np

import tqdm

import os
import sys

import heapq
import math
import time

from home_robot_hw.remote import StretchClient

#import sys
#sys.path.append('/home/hello-robot/hello-robot-peiqi')
#from util import navigate

import yaml
from pathlib import Path

import time

map_data = yaml.safe_load(Path("map_data.yaml").read_text())
xmin, ymin, resolution = map_data['xmin'], map_data['ymin'], map_data['resolution']

robot = StretchClient()

POS_TOL = 0.1
YAW_TOL = 0.2

def neighbors(pt):
    return [(pt[0] + dx, pt[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]

def compute_heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def to_pt(xy):
        return (
            int((xy[0] - xmin) / resolution + 0.5),
            int((xy[1] - ymin) / resolution + 0.5),
        )

def to_xy(pt):
    return (
        pt[0] * resolution + xmin,
        pt[1] * resolution + ymin,
    )

def is_in_line_of_sight(start_pt, end_pt):

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) > abs(dy):
            if dx < 0:
                start_pt, end_pt = end_pt, start_pt
            for x in range(start_pt[0], end_pt[0] + 1):
                yf = start_pt[1] + (x - start_pt[0]) / dx * dy
                # if self.point_is_occupied(x, int(yf)):
                #     return False
                for y in list({math.floor(yf), math.ceil(yf)}):
                    if (x, y) not in valid_pts:
                        return False

        else:
            if dy < 0:
                start_pt, end_pt = end_pt, start_pt
            for y in range(start_pt[1], end_pt[1] + 1):
                xf = start_pt[0] + (y - start_pt[1]) / dy * dx
                # if self.point_is_occupied(int(x), y):
                #     return False
                for x in list({math.floor(xf), math.ceil(xf)}):
                    if (x, y) not in valid_pts:
                        return False

        return True

def get_unoccupied_neighbor(pt, goal_pt):
        neighbor_pts = neighbors(pt)
        if goal_pt is not None:
            goal_pt_non_null = goal_pt
            neighbor_pts.sort(key=lambda n: compute_heuristic(n, goal_pt_non_null))
        for neighbor_pt in neighbor_pts:
            if neighbor_pt in valid_pts:
                return neighbor_pt

def get_reachable_points(start_pt):

        #start_pt = get_unoccupied_neighbor(start_pt)

        reachable_points = set()
        to_visit = [start_pt]
        while to_visit:
            pt = to_visit.pop()
            #print(reachable_points)
            if pt in reachable_points:
                continue
            reachable_points.add(pt)
            for new_pt in neighbors(pt):
                if new_pt in reachable_points:
                    continue
                if not new_pt in valid_pts:
                    continue
                to_visit.append(new_pt)
        return reachable_points


def clean_path(path):
        cleaned_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if is_in_line_of_sight(path[i], path[j]):
                    break
            else:
                j = i + 1
            cleaned_path.append(path[j])
            i = j
        return cleaned_path

def plan(start_xy, end_xy):
    start_pt, end_pt = to_pt(start_xy), to_pt(end_xy)
    q = [(0, start_pt)]
    came_from = {start_pt: None}
    cost_so_far = {start_pt: 0.0}
    while q:
        _, current = heapq.heappop(q)
        if current == end_pt:
            break
            
        for nxt in neighbors(current):
            if nxt not in valid_pts:
                continue
                
            new_cost = cost_so_far[current] + compute_heuristic(current, nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + compute_heuristic(end_pt, nxt)
                heapq.heappush(q, (priority, nxt))
                came_from[nxt] = current
    path = []
    current = end_pt
    while current != start_pt:
        path.append(current)
        prev = came_from[current]
        if prev is None:
            break
        current = prev
    path.append(start_pt)
    path.reverse()
    path = clean_path(path)

    return [start_xy] + [to_xy(pt) for pt in path[1:-1]] + [end_xy]

valid_points = np.load('map.npy')
valid_pts = [to_pt(coord) for coord in valid_points]

def navigate(robot, goal):
    while True:
        robot.nav.navigate_to(xyt_goal)
        xyt_curr = robot.nav.get_base_pose()
        print("The robot currently loactes at " + str(xyt_curr))
        time.sleep(0.5)
        if np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL) and np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL):
            print("The robot is finally at " + str(xyt_goal))
            break

def compute_theta(cur_x, cur_y, end_x, end_y):
    theta = 0
    if end_x == cur_x and end_y >= cur_y:
        theta = np.pi / 2
    elif end_x == cur_x and end_y < cur_y:
        theta = -np.pi / 2
    else:
        theta = np.arctan((end_y - cur_y) / (end_x - cur_x))
        if end_x < cur_x:
            theta = theta + np.pi
    return theta

start_xy = robot.nav.get_base_pose()[:2]
start_pt = to_pt(start_xy)
valid_targets = np.array([to_xy(rp) for rp in get_reachable_points(start_pt)])

while True:
    start_xy = robot.nav.get_base_pose()[:2]
    start_x, start_y = start_xy[0], start_xy[1]
    end_x = float(input("Enter x:"))
    print("end_x =", end_x)
    end_y = float(input("Enter y:"))
    print("end_y =", end_y)
    #end_theta = float(input("Enter theta:"))
    #print("end_theta =", end_theta)
    #valid_targets = [to_xy(rp) for rp in get_reachable_points((end_x, end_y))]
    #print(valid_targets)
    end_xy = valid_targets[np.argmin(np.linalg.norm(valid_targets - [end_x, end_y], axis = -1))]
    #end_xy = valid_targets[np.argmin(
    #    np.linalg.norm(valid_targets - [end_x, end_y], axis = -1) + 0.8 * np.linalg.norm(valid_targets - [start_x, start_y], axis = -1))]
    print("start at ", robot.nav.get_base_pose())
    print(to_pt(start_xy) in valid_pts)
    print(to_pt(end_xy) in valid_pts)
    print(to_pt(end_xy), end_xy)
    print([3.5078125, 1.1515625] in valid_points)
    waypoints = plan(start_xy[:2], end_xy[:2])
    print("path planned is ", waypoints)
    theta = 0
    for i in range(len(waypoints) - 1):
        xyt_curr = robot.nav.get_base_pose()
        if i + 1 != len(waypoints) - 1:
            theta = compute_theta(waypoints[i + 1][0], waypoints[i + 1][1], waypoints[i + 2][0], waypoints[i + 2][1])
        else:
            theta = compute_theta(waypoints[i + 1][0], waypoints[i + 1][1], end_x, end_y)
        xyt_goal = np.array([waypoints[i + 1][0], waypoints[i + 1][1], theta])
        print("The robot is moving to " + str(xyt_goal))
        navigate(robot, xyt_goal)
    #navigate(robot, [end_xy[0], end_xy[1], theta])
        
    
