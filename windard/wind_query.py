import numpy as np
from floris.wind_data import WindDataBase


class WindQuery:
    def __init__(
        self,
        directions=None,
        speeds=None,
        TIs=None,
    ) -> None:

        self.directions = np.array([]) if directions is None else directions
        self.speeds = np.array([]) if speeds is None else speeds
        self.TIs = np.array([]) if TIs is None else TIs
        self.N_conditions = self.directions.size

    def set_directions(self, directions):
        self.directions = directions
        self.N_conditions = (
            None if self.directions.size != self.speeds.size else self.directions.size
        )

    def set_speeds(self, speeds):
        self.speeds = speeds
        self.N_conditions = (
            None if self.directions.size != self.speeds.size else self.directions.size
        )

    def set_TIs(self, TIs):
        # flip over to using a numpy array and correct size if set as a float
        TIs = np.array(TIs)
        if (np.array(TIs).size == 1) and (self.N_conditions is not None):
            TIs = TIs * np.ones((self.N_conditions,))
        else:
            assert (
                TIs.size == self.N_conditions
            ), "mismatch in TI size vs. direction/speed"
        self.TIs = TIs

    def get_directions(self):
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.directions

    def get_speeds(self):
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.speeds

    def get_TIs(self):
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.TIs

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
          - winddata_FLORIS: ... TO DO
        """

        wind_directions, wind_speeds, ti_table, freq_table, _, _ = (
            winddata_FLORIS.unpack()
        )

        # create the wind query for this condition
        wq = WindQuery(wind_directions, wind_speeds, TIs=ti_table)
        assert wq.is_valid()  # make sure it's legit
        return wq  # and ship it
