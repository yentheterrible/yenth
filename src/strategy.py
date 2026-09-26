from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from math_helpers import clamp, normalized_or
from util.orientation import Orientation
from util.vec import Vec3


FIELD_HALF_LENGTH = 5120.0
FIELD_HALF_WIDTH = 4096.0
BALL_RADIUS = 93.0


class BotState(Enum):
    KICKOFF = auto()
    OFFENSE = auto()
    ROTATE_TO_DEFEND = auto()
    SHADOW_DEFENSE = auto()
    RECOVERY = auto()


class Play(Enum):
    KICKOFF = auto()
    SHOT = auto()
    CLEAR = auto()
    CONTROL = auto()
    CHALLENGE = auto()
    SHADOW = auto()
    ROTATE = auto()
    RECOVER = auto()


@dataclass
class Situation:
    self_eta: float
    teammate_eta: float
    opponent_eta: float
    threat: bool
    opponent_touch: bool
    team_first: bool
    opponents: List[Vec3]
    last_defender: Optional[Vec3]
    ball_speed: float


@dataclass
class Plan:
    play: Play
    drive_target: Vec3
    hit_pos: Vec3
    shot_target: Vec3
    speed: float
    boost_reserve: float


class DecisionMaker:
    def __init__(self, team: int, index: int):
        self.team = team
        self.index = index
        self.attack_sign = 1.0 if team == 0 else -1.0

    def analyze(self, packet, ball, prediction, hit_pos: Vec3, now: float) -> Situation:
        self_car = packet.game_cars[self.index]
        self_eta = self._reach_time(self_car, hit_pos)
        teammate_etas = []
        opponent_etas = []
        opponents = []
        last_defender = None
        furthest_progress = -float("inf")

        for index in range(packet.num_cars):
            player = packet.game_cars[index]
            if player.is_demolished:
                continue
            pos = Vec3(player.physics.location)
            if player.team == self.team:
                if index != self.index:
                    teammate_etas.append(self._reach_time(player, hit_pos))
                continue

            opponents.append(pos)
            opponent_etas.append(self._reach_time(player, hit_pos))
            progress = pos.y * self.attack_sign
            if progress > furthest_progress:
                furthest_progress = progress
                last_defender = pos

        touch = ball.latest_touch
        touch_age = now - touch.time_seconds
        opponent_touch = (touch.time_seconds > 0.0 and 0.0 <= touch_age < 1.4
                          and touch.team != self.team)
        teammate_eta = min(teammate_etas, default=float("inf"))
        opponent_eta = min(opponent_etas, default=float("inf"))
        ball_velocity = Vec3(ball.physics.velocity)
        return Situation(
            self_eta=self_eta,
            teammate_eta=teammate_eta,
            opponent_eta=opponent_eta,
            threat=self._goal_threat(ball, prediction, now),
            opponent_touch=opponent_touch,
            team_first=self_eta <= teammate_eta + 0.12,
            opponents=opponents,
            last_defender=last_defender,
            ball_speed=ball_velocity.length(),
        )

    def choose_state(self, kickoff: bool, on_ground: bool, aerial_phase: int,
                     wavedash_active: bool, world: Situation) -> BotState:
        if kickoff:
            return BotState.KICKOFF if world.team_first else BotState.ROTATE_TO_DEFEND
        if on_ground and wavedash_active:
            return BotState.RECOVERY
        if not on_ground and aerial_phase not in (1, 2, 3, 4):
            return BotState.RECOVERY
        if world.threat:
            return BotState.SHADOW_DEFENSE
        if not world.team_first:
            return BotState.ROTATE_TO_DEFEND
        if world.opponent_eta + 0.45 < world.self_eta:
            return BotState.SHADOW_DEFENSE
        if world.opponent_touch and world.opponent_eta + 0.15 < world.self_eta:
            return BotState.SHADOW_DEFENSE
        return BotState.OFFENSE

    def plan(self, state: BotState, car, car_pos: Vec3, ball: Vec3,
             hit_pos: Vec3, world: Situation) -> Plan:
        own_goal = Vec3(0, -self.attack_sign * FIELD_HALF_LENGTH, 0)
        their_goal = Vec3(0, self.attack_sign * FIELD_HALF_LENGTH, 0)
        shot_target = self._shot_target(hit_pos, world)
        boost_cost = self._boost_cost(car_pos, hit_pos)
        boost_reserve = 7.0 + min(38.0, boost_cost * 0.3)

        if state == BotState.KICKOFF:
            return Plan(Play.KICKOFF, hit_pos, hit_pos, their_goal, 2300.0, 0.0)
        if state == BotState.RECOVERY:
            return Plan(Play.RECOVER, car_pos, hit_pos, their_goal, 1000.0, 0.0)
        if state == BotState.ROTATE_TO_DEFEND:
            return Plan(Play.ROTATE, self._rotation_target(ball, own_goal), hit_pos,
                        their_goal, 1750.0, 25.0)
        if state == BotState.SHADOW_DEFENSE:
            if world.threat and world.team_first and world.self_eta <= world.opponent_eta + 0.35:
                nearest_opp = min(world.opponents, key=lambda pos: pos.dist(hit_pos), default=None)
                clear_target = self._clear_target(hit_pos, nearest_opp)
                touch_dir = normalized_or((clear_target - hit_pos).flat(), Vec3(0, self.attack_sign, 0))
                contact = hit_pos - touch_dir * (BALL_RADIUS + 150.0)
                contact = Vec3(contact.x, contact.y, max(17.0, min(hit_pos.z - 45.0, contact.z)))
                return Plan(Play.CLEAR, contact, hit_pos, clear_target, 2150.0, 8.0)
            return Plan(Play.SHADOW, self._shadow_target(ball, own_goal, world), hit_pos,
                        their_goal, 1650.0, 8.0)

        if world.opponent_eta + 0.35 < world.self_eta and not world.threat:
            return Plan(Play.SHADOW, self._shadow_target(ball, own_goal, world), hit_pos,
                        their_goal, 1500.0, 20.0)
        if (not world.team_first or boost_cost > max(0.0, car.boost - 8.0)) and not world.threat:
            return Plan(Play.ROTATE, self._rotation_target(ball, own_goal), hit_pos,
                        their_goal, 1700.0, 25.0)

        nearest_opp = min(world.opponents, key=lambda pos: pos.dist(hit_pos), default=None)
        if world.threat:
            play = Play.CLEAR
            side = 1.0 if nearest_opp is None or hit_pos.x < nearest_opp.x else -1.0
            touch_target = Vec3(clamp(hit_pos.x + side * 1250.0,
                                      -FIELD_HALF_WIDTH + 500, FIELD_HALF_WIDTH - 500),
                                self.attack_sign * (FIELD_HALF_LENGTH - 650.0), 250.0)
        elif (not world.opponent_touch and world.opponent_eta > 0.75
              and hit_pos.z < 480.0 and world.self_eta < world.opponent_eta):
            play = Play.CONTROL
            away = (Vec3(1.0 if hit_pos.x < 0 else -1.0, 0, 0) if nearest_opp is None
                else normalized_or((hit_pos - nearest_opp).flat(), Vec3(1, 0, 0)))
            control_target = hit_pos + away * 1100.0
            touch_target = Vec3(
                clamp(control_target.x, -FIELD_HALF_WIDTH + 500, FIELD_HALF_WIDTH - 500),
                clamp(control_target.y, -FIELD_HALF_LENGTH + 700, FIELD_HALF_LENGTH - 700),
                max(150.0, hit_pos.z),
            )
        elif world.opponent_eta < world.self_eta + 0.3:
            play = Play.CHALLENGE
            touch_target = (shot_target if world.last_defender is None
                            else self._clear_target(hit_pos, nearest_opp))
        else:
            play = Play.SHOT
            touch_target = shot_target

        touch_dir = normalized_or((touch_target - hit_pos).flat(), Vec3(0, self.attack_sign, 0))
        drive_target = hit_pos - touch_dir * (BALL_RADIUS + 150.0)
        drive_target = Vec3(drive_target.x, drive_target.y,
                            max(17.0, min(hit_pos.z - 45.0, drive_target.z)))
        speed = 1900.0 if play in (Play.CONTROL, Play.CHALLENGE) else 2150.0
        return Plan(play, drive_target, hit_pos, touch_target, speed, boost_reserve)

    def _reach_time(self, car, target: Vec3) -> float:
        if car.is_demolished:
            return float("inf")
        pos = Vec3(car.physics.location)
        delta = (target - pos).flat()
        distance = delta.length()
        if distance < 1.0:
            return max(0.08, max(0.0, target.z - pos.z - 120.0) / 900.0)

        forward = normalized_or(Orientation(car.physics.rotation).forward.flat(), Vec3(1, 0, 0))
        facing = forward.dot(delta / distance)
        speed = Vec3(car.physics.velocity).flat().length()
        acceleration = 950.0 + min(speed, 1800.0) * 0.42
        turn_cost = 1.0 + max(0.0, -facing) * 0.65 + max(0.0, 0.25 - facing) * 0.45
        height_cost = max(0.0, target.z - pos.z - 150.0) / 850.0
        if not car.has_wheel_contact:
            turn_cost += 0.25
        return distance * turn_cost / acceleration + height_cost

    def _goal_threat(self, ball, prediction, now: float) -> bool:
        own_goal_y = -self.attack_sign * FIELD_HALF_LENGTH
        velocity = Vec3(ball.physics.velocity)
        pos = Vec3(ball.physics.location)
        if velocity.y * self.attack_sign < -250.0:
            time_to_goal = (own_goal_y - pos.y) / velocity.y
            if 0.0 < time_to_goal < 3.0:
                x_at_goal = pos.x + velocity.x * time_to_goal
                z_at_goal = pos.z + velocity.z * time_to_goal - 325.0 * time_to_goal ** 2
                if abs(x_at_goal) < 1050.0 and -100.0 < z_at_goal < 750.0:
                    return True

        if prediction is None:
            return False
        for ball_slice in prediction.slices:
            if ball_slice.game_seconds > now + 2.5:
                break
            loc = ball_slice.physics.location
            if (abs(loc.y - own_goal_y) < 650.0 and abs(loc.x) < 1050.0
                    and 0.0 < loc.z < 750.0):
                return True
        return False

    def _shot_target(self, hit_pos: Vec3, world: Situation) -> Vec3:
        goal_y = self.attack_sign * FIELD_HALF_LENGTH
        best = Vec3(0, goal_y, 220.0)
        best_score = -float("inf")
        for x in (-720.0, -360.0, 0.0, 360.0, 720.0):
            target = Vec3(x, goal_y, 220.0)
            clearance = min((self._segment_distance(opp, hit_pos, target)
                             for opp in world.opponents), default=2200.0)
            goalie_bonus = 0.0
            if world.last_defender is not None:
                goalie_bonus = min(900.0, abs(x - world.last_defender.x)) * 0.8
            score = clearance - abs(x) * 0.12 + goalie_bonus
            if score > best_score:
                best, best_score = target, score
        return best

    def _segment_distance(self, point: Vec3, start: Vec3, end: Vec3) -> float:
        segment = (end - start).flat()
        length_sq = segment.dot(segment)
        if length_sq < 1.0:
            return point.dist(start)
        offset = (point - start).flat()
        fraction = clamp(offset.dot(segment) / length_sq, 0.0, 1.0)
        closest = start + segment * fraction
        return (point - closest).flat().length()

    def _clear_target(self, hit_pos: Vec3, opponent: Optional[Vec3]) -> Vec3:
        side = 1.0 if opponent is None or hit_pos.x < opponent.x else -1.0
        return Vec3(clamp(hit_pos.x + side * 1500.0,
                  -FIELD_HALF_WIDTH + 400, FIELD_HALF_WIDTH - 400),
                    self.attack_sign * 4400.0, 250.0)

    def _shadow_target(self, ball: Vec3, own_goal: Vec3, world: Situation) -> Vec3:
        line = (ball - own_goal).flat()
        goal_distance = line.length()
        time_edge = clamp(world.opponent_eta - world.self_eta, -0.7, 1.2)
        distance = clamp(1250.0 + world.ball_speed * 0.28 - time_edge * 650.0,
                 850.0, 3000.0)
        distance = min(distance, max(650.0, goal_distance - 400.0))
        point = own_goal + normalized_or(line, Vec3(0, self.attack_sign, 0)) * distance
        return Vec3(clamp(point.x, -FIELD_HALF_WIDTH + 500, FIELD_HALF_WIDTH - 500),
                clamp(point.y, -FIELD_HALF_LENGTH + 700, FIELD_HALF_LENGTH - 700), 0)

    def _rotation_target(self, ball: Vec3, own_goal: Vec3) -> Vec3:
        lateral = clamp(ball.x * 0.32, -1850.0, 1850.0)
        depth = clamp(ball.y * 0.35, -1400.0, 1400.0)
        return Vec3(lateral, own_goal.y + self.attack_sign * 2600.0 + depth, 0)

    def _boost_cost(self, start: Vec3, target: Vec3) -> float:
        distance = start.dist(target)
        height = max(0.0, target.z - start.z)
        return max(0.0, distance - 1000.0) * 0.018 + height * 0.028

