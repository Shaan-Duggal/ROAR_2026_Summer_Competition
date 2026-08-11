import numpy as np
import math


def normalize_rad(rad: float):
    return rad % (2 * np.pi)


class LatController:
    DEFAULT_PURE_PURSUIT_GAIN = 1.5
    DEFAULT_WHEELBASE_M = 4.7

    def __init__(self, pure_pursuit_gain: float = None, wheelbase_m: float = None):
        self.pure_pursuit_gain = (
            self.DEFAULT_PURE_PURSUIT_GAIN if pure_pursuit_gain is None else pure_pursuit_gain
        )
        self.wheelbase_m = (
            self.DEFAULT_WHEELBASE_M if wheelbase_m is None else wheelbase_m
        )

    def run(self, vehicle_location, vehicle_rotation, next_waypoint_location, current_waypoint_idx) -> float:

        waypoint_vector = np.array(next_waypoint_location) - np.array(vehicle_location)

        distance_to_waypoint = np.linalg.norm(waypoint_vector)
        if distance_to_waypoint == 0:
            return 0

        waypoint_vector_normalized = waypoint_vector / distance_to_waypoint

        alpha = normalize_rad(vehicle_rotation[2]) - normalize_rad(
            math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0])
        )
        debug_str = ""
        if 813 < current_waypoint_idx < 840:
            v_angle = normalize_rad(vehicle_rotation[2])
            d_angle = normalize_rad(math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0]))
            debug_str = f"a{alpha} rot{v_angle} {d_angle}"

        steering_command = self.pure_pursuit_gain * math.atan2(
            2.0 * self.wheelbase_m * math.sin(alpha) / distance_to_waypoint, 1.0
        )

        return float(steering_command), debug_str
