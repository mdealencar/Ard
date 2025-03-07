import pytest
import numpy as np
from jax import grad
from jax.test_util import check_grads
import jax.numpy as jnp
import ard.offshore.mooring_constraint as mc

class TestMooringConstraint:
    def setup_method(self):
        pass

class TestDistancePointToMooring:
    def setup_method(self):
        self.distance_point_to_mooring = grad(mc.distance_point_to_mooring, [0])
        pass

    def test_distance_point_to_mooring(self):

        point = np.array([0, 0])

        P_moorings = np.array([[ 10,  10],
                                [ 5,  5],
                                [ 10,  15],
                                [ 15,   5],])

        test_result = mc.distance_point_to_lineseg_nd(point, P_moorings)

        assert test_result == 7.0710678118654755 
