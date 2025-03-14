import pytest
import numpy as np
from jax import grad, jacobian
from jax.test_util import check_grads
import jax.numpy as jnp
import ard.offshore.mooring_constraint as mc

class TestMooringConstraint:
    def setup_method(self):
        pass

class TestConvertInputs_X_Y_To_XY:

    def setup_method(self):
        self.distance_point_to_mooring_grad = grad(mc.convert_inputs_x_y_to_xy, [0])
        pass 

    def test_convert_inputs_x_y_to_xy(self):

        xt_in = np.array([1, 10])
        yt_in = np.array([2, 20])
        xa_in = np.array([[3, 30, 300], [4, 40, 400]])
        ya_in = np.array([[5, 50, 500], [6, 60, 600]])

        pm_out_expected = np.array([[[  1,   2],
                                     [  3,   5],
                                     [ 30,  50],
                                     [300, 500]],
                                    [[ 10,  20],
                                     [  4,   6],
                                     [ 40,  60],
                                     [400, 600]]], dtype=float)
        
        pm_out = mc.convert_inputs_x_y_to_xy(xt_in, yt_in, xa_in, ya_in)

        assert np.all(pm_out == pm_out_expected)

    def test_convert_inputs_x_y_to_xy_grad(self):

        xt_in = jnp.array([1, 10], dtype=float)
        yt_in = jnp.array([2, 20], dtype=float)
        xa_in = jnp.array([[3, 30, 300], [4, 40, 400]], dtype=float)
        ya_in = jnp.array([[5, 50, 500], [6, 60, 600]], dtype=float)

        try:
            check_grads(mc.convert_inputs_x_y_to_xy, (xt_in, yt_in, xa_in, ya_in), order=1)
        except AssertionError:
            pytest.fail("Unexpected AssertionError when checking gradients, gradients may be incorrect")


class TestConvertInputs_X_Y_Z_To_XYZ:

    def setup_method(self):
        self.distance_point_to_mooring_grad = grad(mc.convert_inputs_x_y_z_to_xyz, [0])
        pass 

    def test_convert_inputs_x_y_z_to_xyz(self):

        xt_in = np.array([1, 10], dtype=float)
        yt_in = np.array([2, 20], dtype=float)
        zt_in = np.array([7, 70], dtype=float)
        xa_in = np.array([[3, 30, 300], [4, 40, 400]], dtype=float)
        ya_in = np.array([[5, 50, 500], [6, 60, 600]], dtype=float)
        za_in = np.array([[8, 80, 800], [9, 90, 900]], dtype=float)

        pm_out_expected = np.array([[[  1,   2,   7],
                                     [  3,   5,   8],
                                     [ 30,  50,  80],
                                     [300, 500, 800]],
                                    [[ 10,  20,  70],
                                     [  4,   6,   9],
                                     [ 40,  60,  90],
                                     [400, 600, 900]]], dtype=float)
        
        pm_out = mc.convert_inputs_x_y_z_to_xyz(xt_in, yt_in, zt_in, xa_in, ya_in, za_in)

        assert np.all(pm_out == pm_out_expected)

    def test_convert_inputs_x_y_z_to_xyz_grad(self):

        xt_in = jnp.array([1, 10], dtype=float)
        yt_in = jnp.array([2, 20], dtype=float)
        zt_in = jnp.array([7, 70], dtype=float)
        xa_in = jnp.array([[3, 30, 300], [4, 40, 400]], dtype=float)
        ya_in = jnp.array([[5, 50, 500], [6, 60, 600]], dtype=float)
        za_in = jnp.array([[8, 80, 800], [9, 90, 900]], dtype=float)

        try:
            check_grads(mc.convert_inputs_x_y_z_to_xyz, (xt_in, yt_in, zt_in, xa_in, ya_in, za_in), order=1)
        except AssertionError:
            pytest.fail("Unexpected AssertionError when checking gradients, gradients may be incorrect")

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

class TestDistanceMooringToMooring:
    def setup_method(self):
        self.distance_mooring_to_mooring_grad = grad(mc.distance_mooring_to_mooring, [0])
        self.distance_mooring_to_mooring_jac = jacobian(mc.distance_mooring_to_mooring, [0])
        pass

    def test_distance_mooring_to_mooring_2d_near_end(self):

        P_moorings_A = np.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 20

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(10, rel=1E-3)

    def test_distance_mooring_to_mooring_2d_near_middle(self):

        P_moorings_A = np.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 6
        P_moorings_B[:, 1] -= 10

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        assert float(test_result) == pytest.approx(1.0, rel=1E-2)

    def test_distance_mooring_to_mooring_2d_equal(self):

        P_moorings_A = np.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ])

        P_moorings_B = np.copy(P_moorings_A)

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(0.0, abs=1E-2)

    def test_distance_mooring_to_mooring_3d_near_end(self):

        P_moorings_A = np.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 20

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(10, rel=1E-3)

    def test_distance_mooring_to_mooring_3d_near_middle(self):

        P_moorings_A = np.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ])

        P_moorings_B = np.copy(P_moorings_A)
        P_moorings_B[:, 0] += 6
        P_moorings_B[:, 1] -= 10

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        assert float(test_result) == pytest.approx(1.0, rel=1E-2)

    def test_distance_mooring_to_mooring_3d_equal(self):

        P_moorings_A = np.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ])

        P_moorings_B = np.copy(P_moorings_A)

        test_result = mc.distance_mooring_to_mooring(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert float(test_result) == pytest.approx(0.0, abs=1E-2)

    def test_distance_mooring_to_mooring_2d_near_end_grad(self):

        P_moorings_A = jnp.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ], dtype=float)

        P_moorings_B = jnp.array([[30, 10],
                                [25, 5],
                                [30, 15],
                                [35, 5]
                                ], dtype=float)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(np.array([[0.0, 0.0],
                                                      [0.0, 0.0],
                                                      [0.0, 0.0],
                                                      [-1.0, 0.0]]))

    def test_distance_mooring_to_mooring_2d_near_middle_grad(self):

        P_moorings_A = jnp.array([[100, 100],
                                [50, 50],
                                [100, 150],
                                [150, 50]
                                ], dtype=float)

        P_moorings_B = jnp.array([[160, 0],
                                [110, -50],
                                [160, 100],
                                [210, -50]
                                ], dtype=float)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(np.array([[0.0, 0.0],
                                                      [0.0, 0.0],
                                                      [0.0, 0.0],
                                                      [-1.0, 0.0]]))
        try:
            check_grads(mc.distance_mooring_to_mooring, (P_moorings_A, P_moorings_B), order=1)
        except AssertionError:
            pytest.fail("Unexpected AssertionError when checking gradients, gradients may be incorrect")


    def test_distance_mooring_to_mooring_2d_equal_grad(self):

        P_moorings_A = jnp.array([[10, 10],
                                [5, 5],
                                [10, 15],
                                [15, 5]
                                ], dtype=float)

        P_moorings_B = jnp.copy(P_moorings_A)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(np.array([[0.0, 0.0],
                                                      [0.0, 0.0],
                                                      [0.0, 0.0],
                                                      [0.0, 0.0]]))
        
    def test_distance_mooring_to_mooring_3d_near_end_grad(self):

        P_moorings_A = jnp.array([[10, 10, 0],
                                [5, 5, 0],
                                [10, 15, 0],
                                [15, 5, 0]
                                ], dtype=float)

        P_moorings_B = jnp.array([[30, 10, 0],
                                [25, 5, 0],
                                [30, 15, 0],
                                [35, 5, 0]
                                ], dtype=float)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [-1.0, 0.0, 0.0]]))

    def test_distance_mooring_to_mooring_3d_near_middle_grad(self):

        P_moorings_A = jnp.array([[10, 10, 0.0],
                                [5, 5, 0.0],
                                [10, 15, 0.0],
                                [15, 5, 0.0]
                                ], dtype=float)

        P_moorings_B = jnp.array([[16, 0, 0.0],
                                [11, -5, 0.0],
                                [16, 6, 0.0],
                                [21, -5, 0.0]
                                ], dtype=float)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [-1.0, 0.0, 0.0]]))

        try:
            check_grads(mc.distance_mooring_to_mooring, (P_moorings_A, P_moorings_B), order=1)
        except AssertionError:
            pytest.fail("Unexpected AssertionError when checking gradients, gradients may be incorrect")

    def test_distance_mooring_to_mooring_3d_near_middle_end_grad(self):

        P_moorings_A = jnp.array([[10, 10, 0.0],
                                [5, 5, 0.0],
                                [10, 15, 0.0],
                                [15, 5, 0.0]
                                ], dtype=float)

        P_moorings_B = jnp.array([[16, 0, 0.0],
                                [11, -5, 0.0],
                                [16, 5, 0.0],
                                [21, -5, 0.0]
                                ], dtype=float)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [-1.0, 0.0, 0.0]]))

    def test_distance_mooring_to_mooring_3d_equal_grad(self):

        P_moorings_A = jnp.array([[10, 10, 0.0],
                                [5, 5, 0.0],
                                [10, 15, 0.0],
                                [15, 5, 0.0]
                                ], dtype=float)

        P_moorings_B = jnp.copy(P_moorings_A)

        test_result = self.distance_mooring_to_mooring_grad(P_moorings_A, P_moorings_B)

        # expect error due to smooth functions
        assert test_result[0] == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0]]))