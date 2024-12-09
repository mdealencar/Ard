import os

import numpy as np
import floris

import ard.farm_aero.templates as templates


class FLORISFarmComponent:
    """
    a second-inherit superclass for FLORIS FarmAero components to share common
    code

    TO DO
    """

    def initialize(self):
        self.options.declare("case_title")

    def setup(self):

        # set up FLORIS
        self.fmodel = floris.FlorisModel("defaults")
        self.fmodel.set(
            wind_shear=0.585,
            turbine_type=[self.modeling_options["turbine"]]
        )
        self.fmodel.assign_hub_height_to_ref_height()

        self.case_title = self.options["case_title"]
        self.dir_floris = os.path.join("case_files", self.case_title, "floris_inputs")
        os.makedirs(self.dir_floris, exist_ok=True)

    def compute(self, inputs):

        raise NotImplementedError("compute must be specialized,")

    def setup_partials(self):
        # for FLORIS, no derivatives. use FD because FLORIS is cheap
        self.declare_partials("*", "*", method="fd")

    def get_AEP_farm(self):
        return self.fmodel.get_farm_AEP()

    def get_power_farm(self):
        return self.fmodel.get_farm_power()

    def get_power_turbines(self):
        return self.fmodel.get_turbine_powers().T

    def get_thrust_turbines(self):
        # FLORIS computes the thrust precursors, compute and return thrust
        # use pure FLORIS to get these values for consistency
        CT_turbines = self.fmodel.get_turbine_thrust_coefficients()
        V_turbines = self.fmodel.turbine_average_velocities
        rho_floris = self.fmodel.core.flow_field.air_density
        A_floris = np.pi * self.fmodel.core.farm.rotor_diameters**2 / 4

        thrust_turbines = CT_turbines * (0.5 * rho_floris * A_floris * V_turbines**2)
        return thrust_turbines.T

    def dump_floris_outfile(self, dir_output=None):
        # dump the floris case
        if dir_output is None:
            dir_output = self.dir_floris
        self.fmodel.core.to_file(os.path.join(dir_output, "batch.yaml"))


class FLORISBatchPower(templates.BatchFarmPowerTemplate, FLORISFarmComponent):
    """
    a component to compute a batch power estimate using FLORIS

    ...
    """

    def initialize(self):
        super().initialize()  # run super class script first!
        FLORISFarmComponent.initialize(self)  # FLORIS superclass

    def setup(self):
        super().setup()  # run super class script first!
        FLORISFarmComponent.setup(self)  # setup a FLORIS run

    def setup_partials(self):
        FLORISFarmComponent.setup_partials(self)

    def compute(self, inputs, outputs):

        # generate the list of conditions for evaluation
        self.time_series = floris.TimeSeries(
            wind_directions=np.degrees(np.array(self.wind_query.get_directions())),
            wind_speeds=np.array(self.wind_query.get_speeds()),
            turbulence_intensities=np.array(self.wind_query.get_TIs()),
        )

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x_turbines"],
            layout_y=inputs["y_turbines"],
            wind_data=self.time_series,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
        )
        self.fmodel.set_operation_model("peak-shaving")

        self.fmodel.run()

        # dump the yaml to re-run this case on demand
        FLORISFarmComponent.dump_floris_outfile(self, self.dir_floris)

        # FLORIS computes the powers
        outputs["power_farm"] = FLORISFarmComponent.get_power_farm(self)
        outputs["power_turbines"] = FLORISFarmComponent.get_power_turbines(self)
        outputs["thrust_turbines"] = FLORISFarmComponent.get_thrust_turbines(self)


class FLORISAEP(templates.FarmAEPTemplate):
    """
    a component to compute an AEP estimate using FLORIS

    ...
    """

    def initialize(self):
        super().initialize()  # run super class script first!
        FLORISFarmComponent.initialize(self)  # add on FLORIS superclass

    def setup(self):
        super().setup()  # run super class script first!
        FLORISFarmComponent.setup(self)  # setup a FLORIS run

    def setup_partials(self):
        super().setup_partials()

    def compute(self, inputs, outputs):

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x_turbines"],
            layout_y=inputs["y_turbines"],
            wind_data=self.wind_rose,
            yaw_angles=np.array([inputs["yaw_turbines"]]),
        )
        self.fmodel.set_operation_model("peak-shaving")

        self.fmodel.run()

        # dump the yaml to re-run this case on demand
        FLORISFarmComponent.dump_floris_outfile(self, self.dir_floris)

        # FLORIS computes the powers
        outputs["AEP_farm"] = FLORISFarmComponent.get_AEP_farm(self)
        outputs["power_farm"] = FLORISFarmComponent.get_power_farm(self)
        outputs["power_turbines"] = FLORISFarmComponent.get_power_turbines(self)
        outputs["thrust_turbines"] = FLORISFarmComponent.get_thrust_turbines(self)

    def setup_partials(self):
        FLORISFarmComponent.setup_partials(self)
