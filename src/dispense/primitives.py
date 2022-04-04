import numpy as np

from typing import Union, List


class SinusoidalTrajectory:
    def __init__(
        self,
        time_interval: float,
        axis: Union[List, np.ndarray],
        amplitude: float,
        frequency: float,
    ) -> None:
        assert amplitude > 0, "Invalid amplitude value"
        assert frequency > 0 and frequency < 100, "Invlaid frequency value"
        assert time_interval > 0, "Invalid time interval value"
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.amplitude = amplitude
        self.omega = 2 * np.pi * frequency
        self.time_interval = time_interval
        self.t_step = 0
        self.sum = 0

    def get_twist(self):
        v = (self.amplitude * self.omega) * np.sin(
            self.omega * self.time_interval * self.t_step
        )

        self.sum += v
        v = v * self.axis
        self.t_step += 1
        return v
