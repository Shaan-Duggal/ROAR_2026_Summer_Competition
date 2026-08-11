from functools import reduce
import json
import os
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
import atexit

def dist_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    min_dist = 1000
    min_ind = current_idx
    for i in range(0, 20):
        ind = (current_idx + i) % len(waypoints)
        d = dist_to_waypoint(location, waypoints[ind])
        if d < min_dist:
            min_dist = d
            min_ind = ind
    return min_ind

def findClosestIndex(location, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = dist_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)


class SectionStats:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.section_indeces = []
        self.current_waypoint_idx = 0
        self.num_ticks = 0
        self.section_start_ticks = 0
        self.current_section = 0
        self.lapNum = 1
        self.previous_location = None
        self.section_start_distance = 0
        self.current_distance = 0
        self.initialize()
        
    def initialize(self) -> None:

        sectionLocations = [
            [-278, 372],
            [64, 890],
            [511, 1037],
            [762, 908],
            [198, 307],
            [-11, 60],
            [-85, -339],
            [-210, -1060],
            [-318, -991],
            [-352, -119],
        ]
        self.section_indeces = [2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]


        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

    def step(self) -> None:
        self.num_ticks += 1

        vehicle_location = self.location_sensor.get_last_gym_observation()
        if self.previous_location is not None:
            self.current_distance += np.linalg.norm(vehicle_location - self.previous_location)
        self.previous_location = vehicle_location

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                print(f"Section {i}: ticks {(self.num_ticks - self.section_start_ticks):4d}  distance {(self.current_distance - self.section_start_distance):6.1f}")
                self.section_start_ticks = self.num_ticks
                self.section_start_distance = self.current_distance
                self.current_section = i
                if self.current_section == 0 and self.lapNum != 3:
                    self.lapNum += 1
                    print(f"\nLap {self.lapNum}\n")
