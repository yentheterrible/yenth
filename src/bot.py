import sys
import subprocess

# Self-installing dependency bootstrap
for library in ["rlbot", "numpy", "scipy"]:
    try:
        __import__(library)
    except ImportError:
        print(f"[*] Package missing. Automatically deploying {library} environment...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library, "--quiet"])

import math
from typing import Optional

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from math_helpers import PID, BallPredictor, clamp, normalized_or
from strategy import BotState, DecisionMaker, Play
from util.boost_pad_tracker import BoostPadTracker
from util.orientation import Orientation
from util.vec import Vec3


CEILING_HEIGHT = 2044.0
FIELD_HALF_LENGTH = 5120.0
FIELD_HALF_WIDTH = 4096.0


class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.boost_pad_tracker = BoostPadTracker()
        self.ball_predictor = BallPredictor()
        self.decision_maker = DecisionMaker(team, index)
        self.steer_pid = PID(2.8, 0.0, 0.12)
        self.speed_pid = PID(0.0025, 0.00005, 0.00015, integral_limit=500.0)
        self.state = BotState.KICKOFF
        self.last_time: Optional[float] = None
        self.slide_cooldown = 0.0
        self.slide_burst_remaining = 0.0
        self.aerial_phase = 0
        self.aerial_elapsed = 0.0
        self.kickoff_active = False
        self.was_airborne = False
        self.wavedash_window = 0.0
        self.play = Play.ROTATE

    def initialize_agent(self):
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.boost_pad_tracker.update_boost_status(packet)
        car = packet.game_cars[self.index]
        ball = packet.game_ball
        car_location = Vec3(car.physics.location)
        car_velocity = Vec3(car.physics.velocity)
        ball_location = Vec3(ball.physics.location)
        now = packet.game_info.seconds_elapsed
        dt = 1.0 / 60.0 if self.last_time is None else clamp(now - self.last_time, 1.0 / 120.0, 0.1)
        self.last_time = now
        self.slide_cooldown = max(0.0, self.slide_cooldown - dt)
        self.slide_burst_remaining = max(0.0, self.slide_burst_remaining - dt)
        on_ground = car.has_wheel_contact
        if self.was_airborne and on_ground:
            self.wavedash_window = 0.1
        self.wavedash_window = max(0.0, self.wavedash_window - dt)
        self.was_airborne = not on_ground

        prediction = self.get_ball_prediction_struct()
        hit = self.ball_predictor.find_intercept(
            prediction, now, car_location, car_velocity.length())
        hit_pos = hit[0] if hit is not None else ball_location
        hit_time = hit[1] if hit is not None else 0.0

        if packet.game_info.is_kickoff_pause:
            self.kickoff_active = True
        elif self.kickoff_active and (ball_location.length() > 700.0 or
                                      Vec3(ball.physics.velocity).length() > 500.0):
            self.kickoff_active = False
        kickoff = self.kickoff_active
        world = self.decision_maker.analyze(packet, ball, prediction, hit_pos, now)
        self.state = self.decision_maker.choose_state(
            kickoff, on_ground, self.aerial_phase, self.wavedash_window > 0.0, world)
        plan = self.decision_maker.plan(self.state, car, car_location, ball_location, hit_pos, world)
        self.play = plan.play
        controls = self._drive(car, plan.drive_target, plan.speed, car_velocity.length(), dt)

        if self.state == BotState.RECOVERY:
            self._apply_recovery(car, controls, plan.drive_target)
        elif self._should_fast_aerial(car, plan.hit_pos, hit_time, plan.boost_reserve):
            self._apply_fast_aerial(car, controls, dt, plan.hit_pos,
                                    plan.boost_reserve, world.threat)
        else:
            self.aerial_phase = 0
            self.aerial_elapsed = 0.0

        if self.state == BotState.KICKOFF:
            controls.boost = car.boost > 0 and car_velocity.length() < 2200.0
        elif self.state != BotState.RECOVERY and self._should_boost(
                controls, car, car_location, plan.drive_target, car_velocity.length(),
                plan.boost_reserve, world.threat):
            controls.boost = True

        self.renderer.draw_line_3d(car_location, plan.drive_target, self.renderer.white())
        self.renderer.draw_string_3d(
            car_location, 1, 1,
            f"{self.state.name}/{self.play.name} | {car_velocity.length():.0f}", self.renderer.white())
        self.renderer.draw_rect_3d(plan.drive_target, 8, 8, True, self.renderer.cyan(), centered=True)
        if hit is not None:
            self.renderer.draw_line_3d(ball_location, hit_pos, self.renderer.cyan())
            self.renderer.draw_line_3d(hit_pos, plan.shot_target, self.renderer.orange())
        return controls

    def _drive(self, car, target: Vec3, target_speed: float, speed: float,
               dt: float) -> SimpleControllerState:
        controls = SimpleControllerState()
        orientation = Orientation(car.physics.rotation)
        relative = target - Vec3(car.physics.location)
        flat_target = relative.flat()
        desired = normalized_or(flat_target, orientation.forward.flat())
        forward = normalized_or(orientation.forward.flat(), Vec3(1, 0, 0))
        angle = math.atan2(forward.x * desired.y - forward.y * desired.x,
                           forward.x * desired.x + forward.y * desired.y)
        controls.steer = self.steer_pid.step(angle, dt, angular=True)

        reverse_target = abs(angle) > 2.25 and speed < 900.0
        signed_target_speed = -500.0 if reverse_target else target_speed
        speed_error = signed_target_speed - speed
        controls.throttle = self.speed_pid.step(speed_error, dt)

        if abs(angle) > 2.15 and speed > 900.0 and self.slide_cooldown <= 0.0:
            self.slide_burst_remaining = 0.12
            self.slide_cooldown = 0.45
        controls.handbrake = self.slide_burst_remaining > 0.0
        return controls

    def _should_fast_aerial(self, car, hit_pos: Vec3, hit_time: float,
                            boost_reserve: float) -> bool:
        if self.aerial_phase in (1, 2, 3, 4):
            return True
        height_gap = hit_pos.z - car.physics.location.z
        required_vertical_speed = height_gap / max(hit_time, 0.1)
        vertical_delta = required_vertical_speed - car.physics.velocity.z
        return (car.has_wheel_contact and car.boost > max(20.0, boost_reserve)
                and 0.65 <= hit_time <= 1.8 and height_gap > 550.0
                and 0.0 < required_vertical_speed < 1800.0 and vertical_delta < 1800.0)

    def _apply_fast_aerial(self, car, controls: SimpleControllerState, dt: float,
                           hit_pos: Vec3, boost_reserve: float, emergency: bool) -> None:
        if self.aerial_phase == 4 and car.has_wheel_contact:
            self.aerial_phase = 0
            self.aerial_elapsed = 0.0
            return
        if self.aerial_phase == 0:
            self.aerial_phase = 1
            self.aerial_elapsed = 0.0
        self.aerial_elapsed += dt
        controls.boost = car.boost > max(5.0, boost_reserve) or emergency
        if self.aerial_phase == 1:
            controls.jump = True
            if self.aerial_elapsed >= 0.18:
                self.aerial_phase = 2
                self.aerial_elapsed = 0.0
        elif self.aerial_phase == 2:
            controls.jump = False
            if self.aerial_elapsed >= 0.04:
                self.aerial_phase = 3
                self.aerial_elapsed = 0.0
        elif self.aerial_phase == 3:
            controls.jump = True
            if self.aerial_elapsed >= 0.12:
                self.aerial_phase = 4
        else:
            controls.jump = False
        if not car.has_wheel_contact:
            direction = normalized_or(hit_pos - Vec3(car.physics.location), Orientation(car.physics.rotation).forward)
            self._attitude_controls(car, controls, direction, Vec3(0, 0, 1))

    def _attitude_controls(self, car, controls: SimpleControllerState,
                           desired_forward: Vec3, desired_up: Vec3) -> None:
        orientation = Orientation(car.physics.rotation)
        forward = normalized_or(orientation.forward, Vec3(1, 0, 0))
        right = normalized_or(orientation.right, Vec3(0, 1, 0))
        up = normalized_or(orientation.up, Vec3(0, 0, 1))
        target_forward = normalized_or(desired_forward, forward)
        target_up = normalized_or(desired_up, up)
        angular_velocity = Vec3(car.physics.angular_velocity)
        error = forward.cross(target_forward)
        controls.pitch = clamp(2.4 * error.dot(right) - 0.14 * angular_velocity.dot(right), -1.0, 1.0)
        controls.yaw = clamp(2.4 * error.dot(up) - 0.14 * angular_velocity.dot(up), -1.0, 1.0)
        roll_error = up.cross(target_up).dot(forward)
        controls.roll = clamp(2.0 * roll_error - 0.14 * angular_velocity.dot(forward), -1.0, 1.0)

    def _apply_recovery(self, car, controls: SimpleControllerState, target: Vec3) -> None:
        orientation = Orientation(car.physics.rotation)
        location = Vec3(car.physics.location)
        surface_up = self._surface_normal(location)
        velocity = Vec3(car.physics.velocity)
        landing_forward = velocity - surface_up * velocity.dot(surface_up)
        if landing_forward.length() < 100.0:
            landing_forward = orientation.forward - surface_up * orientation.forward.dot(surface_up)
        controls.boost = False
        controls.throttle = 1.0
        self._attitude_controls(car, controls, landing_forward, surface_up)
        if (car.has_wheel_contact and self.wavedash_window > 0.0 and car.jumped
                and surface_up.z > 0.85 and location.z < 260.0
                and velocity.flat().length() > 500.0):
            controls.jump = True
            controls.pitch = -0.5

    def _surface_normal(self, location: Vec3) -> Vec3:
        surfaces = (
            (abs(location.z), Vec3(0, 0, 1)),
            (abs(CEILING_HEIGHT - location.z), Vec3(0, 0, -1)),
            (abs(FIELD_HALF_WIDTH - abs(location.x)), Vec3(-1 if location.x > 0 else 1, 0, 0)),
            (abs(FIELD_HALF_LENGTH - abs(location.y)), Vec3(0, -1 if location.y > 0 else 1, 0)),
        )
        return min(surfaces, key=lambda surface: surface[0])[1]

    def _should_boost(self, controls: SimpleControllerState, car, car_location: Vec3,
                      target: Vec3, speed: float, reserve: float, emergency: bool) -> bool:
        if (car.boost <= 0 or speed > 2200 or controls.handbrake
                or (car.boost <= reserve and not emergency)):
            return False
        direction = normalized_or((target - car_location).flat(), Vec3(1, 0, 0))
        forward = normalized_or(Orientation(car.physics.rotation).forward.flat(), direction)
        alignment = direction.dot(forward)
        return controls.throttle > 0.8 and alignment > 0.85 and car_location.dist(target) > 1200.0
