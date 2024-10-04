import numpy as np
from floris.wind_data import WindDataBase


class WindQuery:
    def __init__(
        self,
        directions=None,
        speeds=None,
    ) -> None:

        self.directions = np.array([]) if directions is None else directions
        self.speeds = np.array([]) if speeds is None else speeds
        self.N_conditions = self.directions.size

    def set_directions(self, directions):
        self.directions = directions
        self.N_conditions = None if self.directions.size != self.speeds.size else self.directions.size

    def set_speeds(self, speeds):
        self.speeds = speeds
        self.N_conditions = None if self.directions.size != self.speeds.size else self.directions.size

    def get_directions(self):
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.directions

    def get_speeds(self):
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.speeds

    def is_valid(self):
        # first, not a valid query if the directions and speeds aren't specified
        if (self.directions is None) or (self.speeds is None):
            return False
        # next, to be valid the directions and speeds should be the same size and shape
        if not np.all(np.equal(self.directions.shape, self.speeds.shape)):
            return False
        # to be valid, directions should be on [0, 360]
        if np.any((self.directions < 0.0) | (self.directions > 360.0)):
            return False
        # to be valid, velocity should be on [0, +inf)
        if np.any(self.directions < 0.0):
            return False
        return True

    def from_FLORIS_WindData(winddata_FLORIS: WindDataBase):
        """
        create a wind query from a FLORIS WindData object, using all datapoints

        args:
          - winddata_FLORIS:
        """

        wind_directions, wind_speeds, ti_table, freq_table, _, _ = (
            winddata_FLORIS.unpack()
        )

        # create the wind query for this condition
        wq = WindQuery(wind_directions, wind_speeds)
        assert wq.is_valid()  # make sure it's legit
        return wq  # and ship it

