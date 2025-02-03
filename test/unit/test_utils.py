import pytest
import numpy as np
import ard.utils as utils


class TestUtils:

    def setup_method(self):
        pass

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
        Check that the smooth max function returns something greater 
        than the true max when called with similar values.
        """

        test_list = np.array([0, 9.999, 10.00, 3.0])
        test_result = utils.smooth_min(x=test_list)

        assert test_result <= 0.0

    def test_smooth_min_not_close(self):
        """
        Check that the smooth max function returns the true max
        when called with very different values.
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = utils.smooth_min(x=test_list)

        assert test_result == pytest.approx(0, rel=1E-15)
    
    def test_smooth_min_cory_close(self):
        """
        Check that the smooth max function returns something greater 
        than the true max when called with similar values.
        """
        
        test_list = np.array([0, 9.999, 10.00, 0.005])
        test_result = utils.smoothmin(target=test_list)

        assert test_result <= 0.0

    def test_smooth_min_cory_not_close(self):
        """
        Check that the smooth max function returns the true max
        when called with very different values.
        """

        test_list = np.array([0, 5, 10.0, 3.0])
        test_result = utils.smoothmin(target=test_list)

        assert test_result == pytest.approx(0.0, rel=1E-15)
