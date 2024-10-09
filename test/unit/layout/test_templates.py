import numpy as np
import openmdao.api as om

import pytest

import windard.layout.templates as layout_templates


class TestLayoutTemplate:

    def setup_method(self):

        self.modeling_options = {
            "farm": {
                "N_turbines": 25,
            },
        }

        self.model = om.Group()
        self.lt = self.model.add_subsystem(
            "layout",
            layout_templates.LayoutTemplate(modeling_options=self.modeling_options),
        )
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling options has the things we need for the layout
        assert "modeling_options" in [k for k, _ in self.lt.options.items()]
        assert "farm" in self.lt.options["modeling_options"].keys()
        assert "N_turbines" in self.lt.options["modeling_options"]["farm"].keys()

        # make sure that the outputs in the component match what we planned
        output_list = [k for k, v in self.lt.list_outputs()]
        for var_to_check in [
            "x_turbines",
            "y_turbines",
            "spacing_effective_primary",
            "spacing_effective_secondary",
        ]:
            assert var_to_check in output_list

    def test_compute(self):

        # make sure that an attempt to compute on the un-specialized class fails
        with pytest.raises(Exception):
            x_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            y_turbines = 7.0 * 130.0 * np.arange(-2, 2, 1)
            self.prob.set_val("x_turbines", x_turbines)
            self.prob.set_val("y_turbines", y_turbines)
            self.prob.run_model()
