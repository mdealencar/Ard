import numpy as np
from floris.wind_data import WindDataBase
from floris.wind_data import TimeSeries


class WindQuery:
    """
    a class for holding queries of the wind conditions

    this class should hold a series of wind conditions for which farm power will
    be computed
    """

    def __init__(
        self,
        directions=None,
        speeds=None,
        TIs=None,
    ) -> None:

        self.directions = np.array([])
        self.speeds = np.array([])
        self.TIs = np.array([])

        if directions is not None:
            self.set_directions(directions)
        if speeds is not None:
            self.set_speeds(speeds)
        if TIs is not None:
            self.set_TIs(TIs)

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

    def set_TI_using_IEC_method(self):
        assert self.directions is not None, "directions must be set"
        assert self.speeds is not None, "speeds must be set"
        # use a temporary FLORIS time series to get the IEC TIs
        ts_temp = TimeSeries(
            wind_directions=self.directions,
            wind_speeds=self.speeds,
            turbulence_intensities=0.06*np.ones_like(self.directions),  # default value
        )
        ts_temp.assign_TI_using_IEC_method()

        # re-set all the values
        _, _, TIs_new, _, _, _ = ts_temp.unpack()
        # directions_new, speeds_new, TIs_new, _, _, _ = ts_temp.unpack()
        # self.set_directions(directions_new)
        # self.set_speeds(speeds_new)
        self.set_TIs(TIs_new)

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

        wind_directions, wind_speeds, ti_table, _, _, _ = (
            winddata_FLORIS.unpack()
        )

        # create the wind query for this condition
        wq = WindQuery(wind_directions, wind_speeds, TIs=ti_table)
        assert wq.is_valid()  # make sure it's legit
        return wq  # and ship it
