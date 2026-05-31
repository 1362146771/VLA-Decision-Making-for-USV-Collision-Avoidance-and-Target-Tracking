# core/execution/action_executor.py
"""
ActionExecutor: maps discrete action predictions to continuous USV control setpoints.

Implements the action decoding lookup table from Table 3 (Section 4.6):

  Action      | Throttle theta | Rudder angle delta | Control logic
  ------------|----------------|---------------------|---------------
  FORWARD     | 0.5            | 0 deg               | Constant setpoint
  STOP        | 0.0            | 0 deg               | Constant setpoint
  TURNLEFT    | 0.4            | -15 deg             | Constant setpoint
  TURNRIGHT   | 0.4            | +15 deg             | Constant setpoint
  ACCELERATE  | theta + 0.1   | -                   | Incremental update
  DECELERATE  | theta - 0.1   | -                   | Incremental update

Temporal smoothing (Eq. 18):
  u_smooth_t = alpha * u_raw_t + (1 - alpha) * u_smooth_{t-1},   alpha = 0.3
  Time constant tau approx 0.5 s at 2 Hz decision frequency.

Notation:
  theta in [0, 1]        -- throttle command (0 = full stop, 1 = full speed)
  delta in [-30, 30] deg -- rudder angle (negative = port/left, positive = starboard/right)
"""

from typing import Dict, Tuple


# Discrete action index mapping (|A| = 6)
ACTION_NAMES = {
    0: "FORWARD",
    1: "STOP",
    2: "TURNLEFT",
    3: "TURNRIGHT",
    4: "ACCELERATE",
    5: "DECELERATE",
}


class ActionExecutor:
    """
    Converts discrete action indices or names to low-level USV control commands.

    Maintains current throttle state for incremental ACCELERATE/DECELERATE updates
    and applies EMA smoothing to prevent abrupt maneuvers (Section 4.6).

    Args:
        alpha: EMA smoothing coefficient (alpha = 0.3, paper default).
    """

    # Fixed control setpoints (Table 3)
    _SETPOINTS = {
        "FORWARD":    {"throttle": 0.5,  "rudder":  0.0,  "incremental": False},
        "STOP":       {"throttle": 0.0,  "rudder":  0.0,  "incremental": False},
        "TURNLEFT":   {"throttle": 0.4,  "rudder": -15.0, "incremental": False},
        "TURNRIGHT":  {"throttle": 0.4,  "rudder": +15.0, "incremental": False},
        "ACCELERATE": {"throttle": +0.1, "rudder":  None,  "incremental": True},
        "DECELERATE": {"throttle": -0.1, "rudder":  None,  "incremental": True},
    }

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._current_throttle: float = 0.0
        self._smooth_throttle:  float = 0.0
        self._smooth_rudder:    float = 0.0

    def reset(self):
        """Reset internal state at the start of each episode."""
        self._current_throttle = 0.0
        self._smooth_throttle  = 0.0
        self._smooth_rudder    = 0.0

    def to_control(self, action) -> Dict[str, float]:
        """
        Convert a discrete action to raw (unsmoothed) control setpoints.

        Args:
            action: int index (0..5) or action name string
        Returns:
            dict with 'throttle' and 'rudder'
        """
        if isinstance(action, int):
            action = ACTION_NAMES.get(action, "FORWARD")
        action = str(action).upper()
        sp = self._SETPOINTS.get(action, self._SETPOINTS["FORWARD"])

        if sp["incremental"]:
            self._current_throttle = float(
                max(0.0, min(1.0, self._current_throttle + sp["throttle"]))
            )
            return {"throttle": self._current_throttle, "rudder": self._smooth_rudder}
        else:
            self._current_throttle = sp["throttle"]
            return {"throttle": sp["throttle"], "rudder": sp["rudder"]}

    def to_smooth_control(self, action) -> Dict[str, float]:
        """
        Convert action to EMA-smoothed control commands (Eq. 18).

        u_smooth_t = alpha * u_raw_t + (1 - alpha) * u_smooth_{t-1}

        Returns:
            dict with smoothed 'throttle' and 'rudder'
        """
        raw = self.to_control(action)
        a = self.alpha
        self._smooth_throttle = a * raw["throttle"] + (1 - a) * self._smooth_throttle
        self._smooth_rudder   = a * raw["rudder"]   + (1 - a) * self._smooth_rudder
        return {"throttle": self._smooth_throttle, "rudder": self._smooth_rudder}

    def to_tuple(self, action, smooth: bool = True) -> Tuple[float, float]:
        """Return (throttle, rudder) control tuple."""
        ctrl = self.to_smooth_control(action) if smooth else self.to_control(action)
        return ctrl["throttle"], ctrl["rudder"]
