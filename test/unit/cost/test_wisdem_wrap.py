import os

import numpy as np
import openmdao.api as om

from wisdem.inputs.validation import load_yaml

import ard
import ard.utils
import ard.layout.gridfarm as gridfarm
import ard.cost.wisdem_wrap as wcost
import ard.glue.prototype as glue


class TestLandBOSSE:

    def setup_method(self):

        # specify the configuration/specification files to use
        filename_turbine_spec = os.path.abspath(
            os.path.join(
                os.path.split(ard.__file__)[0],
                "..",
                "examples",
                "data",
                "turbine_spec_IEA-3p4-130-RWT.yaml",
            )
        )  # toolset generalized turbine specification
        filename_turbine_FLORIS = os.path.abspath(
            os.path.join(
                os.path.split(ard.__file__)[0],
                "..",
                "examples",
                "data",
                "FLORIS_turbine_library",
                "IEA-3p4-130-RWT.yaml",
            )
        )  # toolset generalized turbine specification
        filename_floris_config = os.path.abspath(
            os.path.join(
                os.path.split(ard.__file__)[0],
                "..",
                "examples",
                "data",
                "FLORIS.yaml",
            )
        )  # default FLORIS config for the project
        # create a FLORIS yaml to conform to the config/spec files above
        ard.utils.create_FLORIS_yamlfile(filename_turbine_spec, filename_turbine_FLORIS)
        # load the turbine specification
        data_turbine = load_yaml(filename_turbine_spec)

        # set up the modeling options
        self.modeling_options = {
            "farm": {
                "N_turbines": 25,
            },
            "turbine": data_turbine,
            "FLORIS": {
                "filename_tool_config": filename_floris_config,
            },
        }

        # create an OM model and problem
        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.landbosse = self.model.add_subsystem(
            "landbosse",
            wcost.LandBOSSE(),
        )
        self.model.connect(  # effective primary spacing for BOS
            "spacing_effective_primary", "landbosse.turbine_spacing_rotor_diameters"
        )
        self.model.connect(  # effective secondary spacing for BOS
            "spacing_effective_secondary", "landbosse.row_spacing_rotor_diameters"
        )

        self.prob = om.Problem(self.model)
        self.prob.setup()

        # setup the latent variables for LandBOSSE and FinanceSE
        wcost.LandBOSSE_setup_latents(self.prob, self.modeling_options)
        # wcost.FinanceSE_setup_latents(self.prob, self.modeling_options)

    def test_baseline_farm(self):

        self.prob.set_val("gridfarm.spacing_primary", 7.0)
        self.prob.set_val("gridfarm.spacing_secondary", 7.0)
        self.prob.set_val("gridfarm.angle_orientation", 0.0)
        self.prob.set_val("gridfarm.angle_skew", 0.0)

        self.prob.run_model()

        ### BEGIN: I THINK THIS SHOULD BE A BROADLY SHARED FUNCTION
        fn_pyrite = os.path.join(
            os.path.split(__file__)[0],
            "test_wisdem_wrap_baseline_farm.npz",
        )
        if False:  # set to True to write new pyrite value file
            # create pyrite file
            print(f"writing new pyrite file in {fn_pyrite}")
            np.savez(
                fn_pyrite,
                bos_capex_kW=np.array(
                    self.prob.get_val("landbosse.bos_capex_kW", units="USD/kW")
                ),
                total_capex=np.array(
                    self.prob.get_val("landbosse.total_capex", units="MUSD")
                ),
            )
            assert False
        else:
            pyrite_data = np.load(fn_pyrite)
            assert np.isclose(
                pyrite_data["bos_capex_kW"],
                self.prob.get_val("landbosse.bos_capex_kW", units="USD/kW"),
            )
            assert np.isclose(
                pyrite_data["total_capex"],
                self.prob.get_val("landbosse.total_capex", units="MUSD"),
            )
        ### END: I THINK THIS SHOULD BE A BROADLY SHARED FUNCTION


class TestPlantFinance:

    def setup_method(self):
        pass


class TestTurbineCapitalCosts:

    def setup_method(self):
        pass


class TestOperatingExpenses:

    def setup_method(self):
        pass
