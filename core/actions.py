from enum import IntEnum
import math

class Action(IntEnum):
    FORWARD = 0
    STOP = 1
    TURNLEFT = 2
    TURNRIGHT = 3
    ACCELERATE = 4
    DECELERATE = 5

ACTION_NAMES = [a.name for a in Action]
ACTION_SCHEMA = "oceanvla.actions.v1"

def action_id(value):
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(6):
        raise ValueError(f"Expected integer action ID 0..5, received {value!r}")
    return value

def mirror_action(value):
    return {2: 3, 3: 2}.get(action_id(value), value)

class ActionExecutor:
    """Pure decoder/EMA; call update at the low-level control rate (10 Hz).

    Accept a new discrete action at decision rate (2 Hz). Incremental speed
    actions are applied once per decision, not on every control tick.
    Proactive radar braking uses EMA; an emergency stop bypasses it.
    This is a simulator adapter, not a safety-certified controller.
    """
    def __init__(self, alpha=0.3):
        if not 0 < alpha <= 1: raise ValueError("alpha must be in (0,1]")
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.throttle = self.rudder = self.smooth_throttle = self.smooth_rudder = 0.0

    def accept(self, action):
        a = Action(action_id(action))
        fixed = {Action.FORWARD: (0.5, 0.0), Action.STOP: (0.0, 0.0),
                 Action.TURNLEFT: (0.4, -15.0), Action.TURNRIGHT: (0.4, 15.0)}
        if a in fixed: self.throttle, self.rudder = fixed[a]
        else: self.throttle = max(0.0, min(1.0, self.throttle + (0.1 if a == Action.ACCELERATE else -0.1)))

    def update(self, *, radar_distance_m=None, radar_enabled=False,
               proactive_m=30.0, emergency_m=20.0, inside_geofence=True,
               seconds_since_action=0.0):
        if radar_enabled:
            if radar_distance_m is None or not math.isfinite(radar_distance_m) or radar_distance_m < 0:
                raise ValueError("Radar-assisted mode requires valid simulator range")
            if not (math.isfinite(proactive_m) and math.isfinite(emergency_m)
                    and 0 < emergency_m < proactive_m):
                raise ValueError("Invalid radar thresholds")
        fail_safe = not inside_geofence or seconds_since_action >= 1.0
        if fail_safe:
            self.smooth_throttle = 0.0
            return {"throttle": 0.0, "rudder_deg": self.smooth_rudder, "override": "failsafe"}
        if radar_enabled and radar_distance_m <= emergency_m:
            self.smooth_throttle = self.smooth_rudder = 0.0
            return {"throttle": 0.0, "rudder_deg": 0.0, "override": "radar_emergency"}
        proactive = radar_enabled and radar_distance_m <= proactive_m
        a = self.alpha
        target_throttle = 0.0 if proactive else self.throttle
        self.smooth_throttle = a*target_throttle + (1-a)*self.smooth_throttle
        self.smooth_rudder = a*self.rudder + (1-a)*self.smooth_rudder
        return {"throttle": self.smooth_throttle, "rudder_deg": self.smooth_rudder,
                "override": "radar_proactive" if proactive else None}
