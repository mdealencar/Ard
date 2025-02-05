import pytest
import numpy as np
from jax import grad
import ard.utils as utils


class TestUtils:

    def setup_method(self):
        pass

    def test_distance_point_to_lineseg_45_deg_with_end(self):

        point = (10, 10)
        line = ((0, 10), (10,0))

        test_result = utils.distance_point_to_lineseg(point[0], point[1], line[0][0], line[0][1], line[1][0], line[1][1], k_logistic=100.0)

        assert test_result == 7.0710678118654755 

    def test_distance_point_to_lineseg_90_deg_with_end(self):

        point = (0, 0)
        line = ((0, 10), (10,10))

        test_result = utils.distance_point_to_lineseg(point[0], point[1], line[0][0], line[0][1], line[1][0], line[1][1], k_logistic=100.0)

        assert test_result == 10.0 

    def test_distance_point_to_lineseg_gt90_deg_with_end(self):

        point = (0, 0)
        line = ((5, 5), (10,5))

        test_result = utils.distance_point_to_lineseg(point[0], point[1], line[0][0], line[0][1], line[1][0], line[1][1], k_logistic=100.0)

        assert test_result == 7.0710678118654755 

    def test_distance_point_to_lineseg_180_deg_with_end(self):

        point = (0, 5)
        line = ((5, 5), (10,5))

        test_result = utils.distance_point_to_lineseg(point[0], point[1], line[0][0], line[0][1], line[1][0], line[1][1], k_logistic=100.0)

        assert test_result == 5.0

    def test_smooth_max_close(self):
        """
        Check that the smooth max function returns something greater 
        than the true max when called with similar values.
        """

        test_list = np.array([0, 9.999, 10.00, 3.0])
        test_result = utils.smooth_max(x=test_list)

        assert test_result >= 10

    def test_smooth_max_not_close(self):
        """
        Check that the smooth max function returns the true max
        when called with very different values.
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = utils.smooth_max(x=test_list)

        assert test_result == 10

    def test_smooth_min_close(self):
        """
        Check that the smooth min function returns something less 
        than the true min when called with similar values.
        """

        test_list = np.array([0, 9.999, 10.00, 3.0])
        test_result = utils.smooth_min(x=test_list)

        assert test_result <= 0.0

    def test_smooth_min_not_close(self):
        """
        Check that the smooth min function returns the true min
        when called with very different values.
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = utils.smooth_min(x=test_list)

        assert test_result == pytest.approx(0, rel=1E-15)

    def test_smooth_max_grad(self):
        """
        Check that the smooth min function is differentiable and the gradient
        is correct
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        smooth_max_grad = grad(utils.smooth_max)
        test_result = smooth_max_grad(test_list)
        
        assert test_result == pytest.approx([0, 0, 1, 0], rel=1E-6)

    def test_smooth_min_grad(self):
        """
        Check that the smooth min function is differentiable and the gradient
        is correct
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        smooth_min_grad = grad(utils.smooth_min)
        test_result = smooth_min_grad(test_list)
        
        assert test_result == pytest.approx([1, 0, 0, 0], rel=1E-6)