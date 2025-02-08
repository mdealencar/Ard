import os

import numpy as np
import openmdao.api as om

import pytest

import ard.utils
import ard.collection.interarray_wrap as ard_inter


class TestInterarrayCollection:

    def setup_method(self):

        # create the farm layout specification
        farm_spec = {}
        farm_spec["xD_farm"], farm_spec["yD_farm"] = [
            7 * v.flatten()
            for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        ]
        farm_spec["x_substations"] = np.array([-500.0, 500.0])
        farm_spec["y_substations"] = np.array([-500.0, 500.0])

        # specify the configuration/specification files to use
        filename_turbine_spec = os.path.abspath(
            os.path.join(
                ard.__path__[0],
                "..",
                "examples",
                "data",
                "turbine_spec_IEA-3p4-130-RWT.yaml",
            )
        )  # toolset generalized turbine specification
        data_turbine_spec = ard.utils.load_turbine_spec(filename_turbine_spec)

        # set up the modeling options
        modeling_options = {
            "farm": {
                "N_turbines": len(farm_spec["xD_farm"]),
                "N_substations": len(farm_spec["x_substations"]),
            },
            "turbine": data_turbine_spec,
        }

        # create the OpenMDAO model
        model = om.Group()
        self.interarray_coll = model.add_subsystem(
            "interarray_coll",
            ard_inter.InterarrayCollection(
                modeling_options=modeling_options,
            ),
        )

        self.prob = om.Problem(model)
        self.prob.setup()

    def test_setup(self):
        """
        make sure the modeling_options has what we need for farmaero
        """

        assert "modeling_options" in [
            k for k, _ in self.interarray_coll.options.items()
        ]

        assert "farm" in self.interarray_coll.options["modeling_options"].keys()
        assert (
            "N_turbines"
            in self.interarray_coll.options["modeling_options"]["farm"].keys()
        )
        assert (
            "N_substations"
            in self.interarray_coll.options["modeling_options"]["farm"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the inputs in the component match what we planned
            input_list = [k for k, v in self.interarray_coll.list_inputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "x_substations",
                "y_substations",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.interarray_coll.list_outputs()]
            for var_to_check in [
                "length_cables",
                "load_cables",
            ]:
                assert var_to_check in output_list

    def test_compute(self):

        raise NotImplementedError("IMPLEMENT ME!!!!! -cfrontin")
