from typing import Union

import numpy as np

from floris.wind_data import WindDataBase
from floris.wind_data import TimeSeries


class WindQuery:
    """
    A class that manages sets of wind conditions to query that can operate
    for different types and fidelities of aerodynamic solvers without loss of
    generality.
    """

    def __init__(
        self,
        directions: np.ndarray = None,
        speeds: np.ndarray = None,
        TIs: np.ndarray = None,
    ) -> None:
        """
        WindQuery initialization.

        A WindQuery object must be initialized with speeds and directions, which
        must be equal-length numpy arrays, and turbuence intensities can be
        provided with a same-length array, a single float, or will otherwise be
        set by default using FLORIS's `assign_ti_using_IEC_method`.

        Parameters
        ----------
        directions : np.ndarray
            the directions of the wind resource to be evaluated in degrees
        speeds : np.ndarray
            the velocities of the wind resource to be evaluated in meters/second
        TIs : np.ndarray, optional
            turbulence intensities of the wind resource (non-dimensional)
        """

        self.directions = np.array([])
        self.speeds = np.array([])
        self.TIs = np.array([])

        if directions is not None:
            self.set_directions(directions)
        if speeds is not None:
            self.set_speeds(speeds)
        if TIs is not None:
            self.set_TIs(TIs)

    def set_directions(self, directions: np.ndarray):
        """
        Set the directions on a WindQuery object.

        Parameters
        ----------
        directions : np.ndarray
            the directions of the wind resource to be assigned, in degrees
        """

        self.directions = directions
        self.N_conditions = (
            None if self.directions.size != self.speeds.size else self.directions.size
        )

    def set_speeds(self, speeds: np.ndarray):
        """
        Set the wind speeds on a WindQuery object.

        Parameters
        ----------
        speeds : np.ndarray
            the speeds of the wind resource to be assigned, in meters/second
        """

        self.speeds = speeds
        self.N_conditions = (
            None if self.directions.size != self.speeds.size else self.directions.size
        )

    def set_TIs(self, TIs: Union[float, np.ndarray]):
        """
        Set the turbulence intensities on a WindQuery object.

        Parameters
        ----------
        TIs : float or np.ndarray
            the turbulence intensity value or values of the wind resource to be
            assigned, non-dimensionally
        """

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
        """
        Re-set the turbulence intensities using the FLORIS IEC method interface.
        """

        assert self.directions is not None, "directions must be set"
        assert self.speeds is not None, "speeds must be set"
        # use a temporary FLORIS time series to get the IEC TIs
        ts_temp = TimeSeries(
            wind_directions=self.directions,
            wind_speeds=self.speeds,
            turbulence_intensities=0.06
            * np.ones_like(self.directions),  # default value
        )
        ts_temp.assign_ti_using_IEC_method()

        # re-set all the values
        _, _, TIs_new, _, _, _ = ts_temp.unpack()
        self.set_TIs(TIs_new)

    def get_directions(self):
        """Get the directions from the wind query object."""
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.directions

    def get_speeds(self):
        """Get the wind speeds from the wind query object."""
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.speeds

    def get_TIs(self):
        """Get the turbulence intensities from the wind query object."""
        assert self.is_valid(), "mismatch in direction/speed vectors"
        return self.TIs

    def is_valid(self):
        """Ensure that the specified wind conditions are valid."""

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
        Turn a FLORIS WindData object into a (more general) WindQuery object.

        Parameters
        ----------
        winddata_FLORIS : floris.wind_data.WindDataBase
            A FLORIS wind data object derived from the WindDataBase base class.

        Returns
        -------
        WindQuery
            A WindQuery object that represents that same wind data as the FLORIS
            wind data object.
        """

        wind_directions, wind_speeds, ti_table, _, _, _ = winddata_FLORIS.unpack()

        # create the wind query for this condition
        wq = WindQuery(wind_directions, wind_speeds, TIs=ti_table)
        assert wq.is_valid()  # make sure it's legit
        return wq  # and ship it
