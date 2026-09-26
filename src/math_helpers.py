import math
from typing import Optional, Tuple

from util.vec import Vec3


MAX_SPEED = 2300.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_or(vector: Vec3, fallback: Vec3) -> Vec3:
    length = vector.length()
    return vector / length if length > 1e-5 else fallback


class PID:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.previous_error: Optional[float] = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def step(self, error: float, dt: float, angular: bool = False) -> float:
        if dt <= 0.0:
            return clamp(self.kp * error, -1.0, 1.0)

        self.integral = clamp(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        delta = 0.0 if self.previous_error is None else error - self.previous_error
        if angular:
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi

        derivative = delta / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(output, -1.0, 1.0)


class BallPredictor:
    def find_intercept(
        self,
        prediction,
        now: float,
        car_pos: Vec3,
        car_speed: float,
    ) -> Optional[Tuple[Vec3, float]]:
        if prediction is None or prediction.num_slices == 0:
            return None

        best_hit = None
        best_cost = float("inf")
        for ball_slice in prediction.slices:
            hit_time = ball_slice.game_seconds - now
            if hit_time < 0.45:
                continue
            if hit_time > 3.0:
                break

            ball_pos = Vec3(ball_slice.physics.location)
            distance = car_pos.dist(ball_pos)
            reachable = min(MAX_SPEED, car_speed + 1200.0) * hit_time
            reachable += 0.5 * 900.0 * hit_time * hit_time
            height_cost = max(0.0, ball_pos.z - 350.0) * 0.55
            if distance > reachable + 250.0 + height_cost:
                continue

            cost = hit_time + distance / 5000.0 + max(0.0, ball_pos.z - 500.0) / 5000.0
            if cost < best_cost:
                best_hit = (ball_pos, hit_time)
                best_cost = cost
        return best_hit