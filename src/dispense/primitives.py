import numpy as np

from typing import Union, List


class SinusoidalTrajectory:
    def __init__(
        self,
        time_interval: float,
        axis: Union[List, np.ndarray],
        amplitude: float,
        frequency: float,
        pattern: List = [1]
    ) -> None:
        assert amplitude > 0, "Invalid amplitude value"
        assert frequency > 0 and frequency < 100, "Invlaid frequency value"
        assert time_interval > 0, "Invalid time interval value"
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.amplitude = amplitude
        self.omega = 2 * np.pi * frequency
        self.t_period = 1 / frequency
        self.time_interval = time_interval
        self.t_step = 0
        self.pattern = pattern
        self.last_v = 0

    def get_twist(self) -> np.ndarray:
        t = self.time_interval * self.t_step
        if self.pattern[int((t // self.t_period) % len(self.pattern))] == 1:
            v = (self.amplitude * self.omega) * np.sin(self.omega * t)
        else:
            v = 0
        self.t_step += 1
        self.last_v = v
        return v * self.axis
