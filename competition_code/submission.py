from collections import deque
from functools import reduce
import json
import os
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
from WaypointLine import WaypointLine
from SectionStats import SectionStats
import atexit


useDebug = False
useDebugPrinting = False
debugData = {}
dbg_carLocations = []
dbg_wpsToFollow = []
dbg_str = []
dbg_str2 = []
dbg_steer = []


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


def saveDebugData():
    print("Saving...")
    fname = "\\debugData\\line.txt"
    with open(
        f"{os.path.dirname(__file__)}{fname}", "w+"
    ) as outfile:
        outfile.write("\n--- Debug steer\n")
        for line in dbg_steer:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Locatons\n")
        for line in dbg_carLocations:
            outfile.write(f"{line}\n")
        outfile.write("\n--- wpsToFollow\n")
        for line in dbg_wpsToFollow:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Debug str\n")
        for line in dbg_str2:
            outfile.write(f"{line}\n")
        outfile.write("\n--- More Debug str\n")
        for line in dbg_str:
            outfile.write(f"{line}\n")
    print(f"Saved. {fname}")

    if useDebug:
        print("Saving debug data")
        jsonData = json.dumps(debugData, indent=4)
        with open(
            f"{os.path.dirname(__file__)}\\debugData\\debugData.json", "w+"
        ) as outfile:
            outfile.write(jsonData)
        print("Debug Data Saved")


RESPAWN_JUMP_THRESHOLD_M = 20.0


def section_for_waypoint_index(index, section_indeces, wp_count):
    n = len(section_indeces)
    index %= wp_count
    for i, start in enumerate(section_indeces):
        span = (section_indeces[(i + 1) % n] - start) % wp_count
        if (index - start) % wp_count < span:
            return i
    return 0


def average_window_weights(n, ramp):
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    if not ramp:
        return [1.0] * n
    span = float(n - 1)
    return [1.0 + ramp * (i / span - 0.5) for i in range(n)]


def weighted_mean_location(locations, ramp):
    n = len(locations)
    if n == 0:
        return None
    if not ramp:
        total = locations[0]
        for loc in locations[1:]:
            total = total + loc
        return total / n

    weights = average_window_weights(n, ramp)
    total = locations[0] * weights[0]
    for loc, w in zip(locations[1:], weights[1:]):
        total = total + loc * w
    return total / sum(weights)


def resolve_section_indeces(shipped, overrides):
    out = [int(v) for v in shipped]
    for section, value in (overrides or {}).items():
        if value is None:
            continue
        out[int(section)] = int(value)
    return out


def resync_section_after_teleport(
    previous_location,
    current_location,
    waypoint_idx,
    section_indeces,
    wp_count,
    threshold_m=RESPAWN_JUMP_THRESHOLD_M,
):
    if previous_location is None:
        return None
    dx = current_location[0] - previous_location[0]
    dy = current_location[1] - previous_location[1]
    if math.hypot(dx, dy) < threshold_m:
        return None
    return section_for_waypoint_index(waypoint_idx, section_indeces, wp_count)


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_controller = LatController()
        self.throttle_controller = ThrottleController()
        self.section_stats = None
        self.section_indeces = []
        self.section_indeces_override = {s: None for s in range(10)}
        self.num_ticks = 0
        self.current_section = 0
        self.lapNum = 1
        self.previous_waypoint_to_follow = None
        self.max_radius = 10000
        self.previous_location = None
        self.respawn_resync = True
        self.respawn_jump_threshold_m = RESPAWN_JUMP_THRESHOLD_M
        self.respawn_resync_location = None
        self.respawn_resync_count = 0
        self.trace_positions = False
        self.trace_every = 1
        self.lookahead_speed_bounds = [90, 110, 130, 160, 180, 200, 250, 300]
        self.lookahead_values = [9, 11, 14, 18, 22, 26, 30, 35]
        self.lookahead_fallback = 8
        self.lookahead_section_mult = {s: 1.0 for s in range(10)}
        self.avg_num_points_mult = {s: 2.0 for s in range(10)}
        self.avg_num_points_mult.update({0: 1.5, 4: 1.0, 5: 1.0, 6: 1.0,
                                         7: 1.25, 9: 0.0})
        self.avg_num_points_add = {s: 0 for s in range(10)}
        self.avg_num_points_add[4] = 5
        self.avg_num_points_const = {s: None for s in range(10)}
        self.avg_num_points_const[3] = 35
        self.avg_index_offset = {s: None for s in range(10)}
        self.avg_index_offset.update({3: 22, 4: 24, 6: 28})
        self.avg_weight_ramp = 0.0
        self.max_shift_distance_m = {s: 2.0 for s in range(10)}
        self.max_shift_distance_m[1] = 0.2
        self.num_points_before_lookahead = 9
        self.total_dist = 0
        self.waypoint_line = WaypointLine()
        self.previous_brake = False
        self.s3_mult = 1
        self.launch_ticks = 0
        self.launch_throttle = 1.0
        self.launch_brake = 0.0
        self.launch_hand_brake = 0
        self.line_offset_m = {s: 0.0 for s in range(10)}
        self.snap_to_line = {s: (s not in (0, 9)) for s in range(10)}
        self.steer_divisor = 120.0
        self.steer_mult_s2 = 1.2
        self.s3_gate_wps = [800, 801]
        self.s3_gate_reset_wps = [802, 803, 804]
        self.s3_mult_default = 0.85
        self.s3_mult_fast = 0.95
        self.s3_mult_slow = 0.75
        self.s3_brake_speed_kmh = 162.0
        self.s3_slow_speed_kmh = 160.0
        self.s3_wp_gate_early = 813
        self.s3_wp_gate_late = 845
        self.steer_mult_s3_mid = 1.45
        self.steer_mult_s4 = 1.65
        self.steer_cap_s4 = 1.6
        self.steer_mult_s5 = 1.1
        self.steer_mult_s6 = 3.2
        self.steer_clip_lo_s6 = 3.1
        self.steer_clip_hi_s6 = 7.0
        self.steer_mult_s7 = 1.75
        self.s9_wp_gate = 2580
        self.steer_floor_s9_late = 1.7
        self.steer_floor_s9_early = 1.7
        self.s3_clamp_wp_lo = 820
        self.s3_clamp_wp_hi = 837
        self.s3_clamp_steer_lo = -0.007

    async def initialize(self) -> None:
        _npz = np.load(os.path.join(
            os.path.dirname(__file__), "waypoints", "waypointsPrimary.npz"))
        _waypoint_dict = {key: _npz[key] for key in _npz.files}
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(_waypoint_dict)[35:]
        )
        self.section_stats = SectionStats(
            self.maneuverable_waypoints, self.location_sensor, self.velocity_sensor)

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
        self.section_indeces = resolve_section_indeces(
            self.section_indeces, self.section_indeces_override)

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_indeces}")
        print("\nLap 1\n")

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        self.previous_location = vehicle_location


    def _trace_position(self, tick, index, section, x, y, speed_kmh) -> None:
        if not self.trace_positions:
            return
        if self.trace_every > 1 and tick % self.trace_every != 0:
            return
        print("TRACE t={0} i={1} s={2} x={3:.4f} y={4:.4f} v={5:.3f}".format(
            int(tick), int(index), int(section), float(x), float(y),
            float(speed_kmh)))

    def _maybe_resync_section(self, vehicle_location) -> bool:
        if not self.respawn_resync:
            return False

        section = resync_section_after_teleport(
            self.respawn_resync_location,
            vehicle_location,
            self.current_waypoint_idx,
            self.section_indeces,
            len(self.maneuverable_waypoints),
            self.respawn_jump_threshold_m,
        )
        self.respawn_resync_location = vehicle_location
        if section is None:
            return False

        self.current_section = section
        line = getattr(self, "waypoint_line", None)
        if line is not None:
            line.resync_to_location(vehicle_location)
        throttle = getattr(self, "throttle_controller", None)
        if throttle is not None and hasattr(throttle, "reset_after_teleport"):
            throttle.reset_after_teleport()
        self.previous_brake = False
        self.s3_mult = 1
        self.respawn_resync_count += 1
        return True

    async def step(self) -> None:
        self.num_ticks += 1
        self.section_stats.step()

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        self._maybe_resync_section(vehicle_location)

        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                self.current_section = i
                if self.current_section == 0 and self.lapNum != 3:
                    self.lapNum += 1

        self._trace_position(
            self.num_ticks, self.current_waypoint_idx, self.current_section,
            vehicle_location[0], vehicle_location[1], current_speed_kmh)

        nextWaypointIndex = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh, vehicle_location)
        waypoint_to_follow_location = waypoint_to_follow.location
        snap_to_line_location = self.waypoint_line.get_next_waypoint_location(waypoint_to_follow.location)
        _line_offset = self.line_offset_m.get(self.current_section, 0.0)
        if _line_offset:
            snap_to_line_location = self.waypoint_line.offset_location(
                snap_to_line_location, _line_offset)
        if self.snap_to_line.get(self.current_section, True):
            waypoint_to_follow_location = snap_to_line_location

        steer_control, steer_debug = self.lat_controller.run(
            vehicle_location, vehicle_rotation, waypoint_to_follow_location, self.current_waypoint_idx
        )

        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            nextWaypointIndex : nextWaypointIndex + 300
        ]
        num_points_before_lookahead = self.num_points_before_lookahead
        wp_len = len(self.maneuverable_waypoints)
        wp_ind_for_throttle = ((nextWaypointIndex + wp_len) - num_points_before_lookahead) % wp_len
        additional_waypoints = (self.maneuverable_waypoints * 2)[
            wp_ind_for_throttle : wp_ind_for_throttle + 300
        ]
        throttle, brake, gear, speed_data, throttle_debug_str = self.throttle_controller.run(
            waypoints_for_throttle,
            vehicle_location,
            current_speed_kmh,
            self.current_section,
            additional_waypoints,
        )

        if self.current_waypoint_idx in self.s3_gate_wps:
            self.s3_mult = self.s3_mult_default
            if current_speed_kmh >= self.s3_brake_speed_kmh:
                self.s3_mult = self.s3_mult_fast
                if not self.previous_brake:
                    throttle = 0
                    brake = 1
                    self.previous_brake = True
            if current_speed_kmh < self.s3_slow_speed_kmh:
                self.s3_mult = self.s3_mult_slow
            print(f"spd {current_speed_kmh} mult{self.s3_mult} sec={self.current_section}")
        if self.current_waypoint_idx in self.s3_gate_reset_wps:
            self.previous_brake = False

        steerMultiplier = self._steer_multiplier(current_speed_kmh)
        steer_value = self._steer_value(steer_control, steerMultiplier)
        if self.current_waypoint_idx in [2381, 2382] and current_speed_kmh > 257:
            if not self.previous_brake:
              throttle = 0
              brake = 1
              self.previous_brake = True
        if self.current_waypoint_idx in [2383, 2384, 2385]:
            self.previous_brake = False

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": steer_value,
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,
        }

        if self.num_ticks <= self.launch_ticks:
            control["throttle"] = float(np.clip(self.launch_throttle, 0, 1))
            control["brake"] = float(np.clip(self.launch_brake, 0, 1))
            control["hand_brake"] = int(self.launch_hand_brake)
        
        if useDebug:
            dbg_carLocations.append(f"{vehicle_location[0]}, {vehicle_location[1]}")
            dbg_wpsToFollow.append(f"{waypoint_to_follow_location[0]}, {waypoint_to_follow_location[1]}")

            self.total_dist += np.linalg.norm(vehicle_location - self.previous_location)
            self.previous_location = vehicle_location
            s = f"{self.total_dist:.0f}, {current_speed_kmh:.0f}, {speed_data.recommended_speed_now:.0f}, {speed_data.name}, {brake*10:.2f}"
            dbg_str.append(s)
            wp_ind = (self.lapNum-1)*3000 + self.current_waypoint_idx
            s = f"{wp_ind:.0f}, {current_speed_kmh:.0f}, {speed_data.recommended_speed_now:.0f}, {speed_data.name}, {brake*10:.2f}"
            dbg_steer.append(s)

            wpl = waypoint_to_follow_location
            d = np.linalg.norm(waypoint_to_follow.location - vehicle_location)
            s = f"d {self.total_dist:.0f} t {self.num_ticks} ind {self.current_waypoint_idx} \
sp {current_speed_kmh:.2f} rec {speed_data.recommended_speed_now:.1f} dif {(current_speed_kmh - speed_data.recommended_speed_now):.1f} \
r={speed_data.r:.0f}: {throttle_debug_str}, \
t {control['throttle']:.3f} \
br {control['brake']:.3f} \
st: {control['steer']:.10f}, \
{steer_control:.6f}, {steerMultiplier:.6f} trgt wp:ind {nextWaypointIndex} {nextWaypointIndex - self.current_waypoint_idx} {d:.1f} \
loc: ({vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}) wp({wpl[0]:.1f}, {wpl[1]:.1f}) {steer_debug} section {self.current_section}"
            dbg_str2.append(s)


        if useDebug:
            debugData[self.num_ticks] = {}
            debugData[self.num_ticks]["loc"] = [
                round(vehicle_location[0].item(), 3),
                round(vehicle_location[1].item(), 3),
            ]
            debugData[self.num_ticks]["throttle"] = round(float(control["throttle"]), 3)
            debugData[self.num_ticks]["brake"] = round(float(control["brake"]), 3)
            debugData[self.num_ticks]["steer"] = round(float(control["steer"]), 10)
            debugData[self.num_ticks]["speed"] = round(current_speed_kmh, 3)
            debugData[self.num_ticks]["lap"] = self.lapNum


        await self.vehicle.apply_action(control)
        return control

    def _steer_multiplier(self, current_speed_kmh: float):
        steerMultiplier = round((current_speed_kmh + 0.001) / self.steer_divisor, 3)

        if self.current_section == 2:
            steerMultiplier *= self.steer_mult_s2
        if self.current_section in [3]:
            if self.current_waypoint_idx < self.s3_wp_gate_early:
                steerMultiplier *= self.s3_mult
            elif self.current_waypoint_idx < self.s3_wp_gate_late:
                steerMultiplier *= self.steer_mult_s3_mid
            else:
                steerMultiplier *= 1
                self.s3_mult = 1

        if self.current_section == 4:
            steerMultiplier = min(self.steer_cap_s4, steerMultiplier * self.steer_mult_s4)
        if self.current_section == 5:
            steerMultiplier *= self.steer_mult_s5
        if self.current_section in [6]:
            steerMultiplier = np.clip(
                steerMultiplier * self.steer_mult_s6,
                self.steer_clip_lo_s6,
                self.steer_clip_hi_s6,
            )
        if self.current_section == 7:
            steerMultiplier *= self.steer_mult_s7

        if self.current_section == 9:
            if self.current_waypoint_idx > self.s9_wp_gate:
                steerMultiplier = max(steerMultiplier, self.steer_floor_s9_late)
            else:
                steerMultiplier = max(steerMultiplier, self.steer_floor_s9_early)

        return steerMultiplier

    def _steer_value(self, steer_control: float, steerMultiplier: float):
        steer_value = np.clip(steer_control * steerMultiplier, -1, 1)
        if self.s3_clamp_wp_lo < self.current_waypoint_idx < self.s3_clamp_wp_hi:
            steer_value = np.clip(
                steer_control * steerMultiplier, self.s3_clamp_steer_lo, 1)
        return steer_value

    def get_lookahead_value(self, speed):


        base = self.lookahead_fallback
        for speed_upper_bound, num_points in zip(
                self.lookahead_speed_bounds, self.lookahead_values):
            if speed < speed_upper_bound:
                base = num_points
                break

        mult = self.lookahead_section_mult.get(self.current_section, 1.0)
        if mult == 1.0:
            return base
        return max(1, int(round(base * mult)))

    def _avg_num_points(self, section, lookahead_value):
        const = self.avg_num_points_const.get(section)
        if const is not None:
            return const
        mult = self.avg_num_points_mult.get(section, 2.0)
        add = self.avg_num_points_add.get(section, 0)
        return round(lookahead_value * mult) + add

    def _max_shift_distance(self, section):
        return self.max_shift_distance_m.get(section, 2.0)

    def get_lookahead_index(self, speed):
        num_waypoints = self.get_lookahead_value(speed)
        return (self.current_waypoint_idx + num_waypoints) % len(
            self.maneuverable_waypoints
        )


    def next_waypoint_smooth(self, current_speed: float, vehicle_location: float):
        if self.current_section == 3:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 44, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [5, 7]:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [6]:
            kdd = 0.28
            distance = kdd * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]

        return target_waypoint

    def new_RoarPyWaypoint(self, location):
        return roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=12.0)


    def average_point(self, current_speed):
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = self._avg_num_points(self.current_section, lookahead_value)
        _index_offset = self.avg_index_offset.get(self.current_section)
        if _index_offset is not None:
            next_waypoint_index = self.current_waypoint_idx + _index_offset

        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

        next_waypoint_index = next_waypoint_index % len(self.maneuverable_waypoints)
        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location

        sample_points = [
            (start_index_for_avg + i) % len(self.maneuverable_waypoints)
            for i in range(0, num_points)
        ]
        if num_points > 3:
            num_points = len(sample_points)
            new_location = weighted_mean_location(
                [self.maneuverable_waypoints[i].location for i in sample_points],
                self.avg_weight_ramp,
            )
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = self._max_shift_distance(self.current_section)
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv * max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(
                location=new_location,
                roll_pitch_yaw=np.ndarray([0, 0, 0]),
                lane_width=0.0,
            )

        else:
            target_waypoint = self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint
