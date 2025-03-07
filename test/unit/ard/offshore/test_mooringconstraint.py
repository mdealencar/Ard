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
        self.distance_point_to_mooring_grad = grad(mc.distance_point_to_mooring, [0])
        pass

    def test_distance_point_to_mooring_2d_near_end(self):

        point = jnp.array([0, 0])

        P_moorings = jnp.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(7.0710678118654755)

    def test_distance_point_to_mooring_2d_near_middle(self):

        point = jnp.array([7.0, 7.5])

        P_moorings = jnp.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(0.3535533905932738)

    def test_distance_point_to_mooring_3d_near_end_3d(self):

        point = jnp.array([0, 0, 0])

        P_moorings = jnp.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(7.0710678118654755)

    def test_distance_point_to_mooring_3d_near_middle(self):

        point = jnp.array([7.0, 7.5, 0])

        P_moorings = jnp.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ])

        test_result = mc.distance_point_to_mooring(point, P_moorings)

        assert float(test_result) == pytest.approx(0.3535533905932738)

    def test_distance_point_to_mooring_2d_near_end_grad(self):

        point = jnp.array([0, 5], dtype=float)

        P_moorings = jnp.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ], dtype=float)

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)
        
        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0]))

    def test_distance_point_to_mooring_2d_near_middle_grad(self):

        point = jnp.array([9, 13], dtype=float)

        P_moorings = jnp.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ], dtype=float)

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)

        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0]))

    def test_distance_point_to_mooring_3d_near_end_grad(self):

        point = jnp.array([0, 5, 0], dtype=float)

        P_moorings = jnp.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ], dtype=float)

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)
        
        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0, 0.0]))

    def test_distance_point_to_mooring_3d_near_middle_grad(self):

        point = jnp.array([9, 13, 0], dtype=float)

        P_moorings = jnp.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ], dtype=float)

        test_result = self.distance_point_to_mooring_grad(point, P_moorings)

        assert test_result[0] == pytest.approx(np.array([-1.0, 0.0, 0.0]))