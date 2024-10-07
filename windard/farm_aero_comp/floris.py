import os

import numpy as np
import floris

import windard.farm_aero_comp.templates as templates


class FLORISBatchPower(templates.BatchFarmPowerTemplate):
    def initialize(self):
        super().initialize()  # run super class script first!

        self.options.declare("case_title")

    def setup(self):
        super().setup()  # run super class script first!
        # pull out variables from super class
        modeling_options = self.modeling_options
        N_turbines = self.N_turbines

        self.tool_config = modeling_options["FLORIS"]["filename_tool_config"]

        # set up FLORIS
        self.fmodel = floris.FlorisModel(self.tool_config)

        self.case_title = self.options["case_title"]
        self.dir_floris = os.path.join("case_files", self.case_title, "floris_inputs")
        os.makedirs(self.dir_floris, exist_ok=True)

    def setup_partials(self):
        super().setup_partials()

    def compute(self, inputs, outputs):

        # generate the list of conditions for evaluation
        self.time_series = floris.TimeSeries(
            wind_directions=np.degrees(np.array(self.wind_query.get_directions())),
            wind_speeds=np.array(self.wind_query.get_speeds()),
            turbulence_intensities=0.06,  # dummy default for now
        )
        # use IEC 61400-1 standard for TI
        self.time_series.assign_ti_using_IEC_method()

        # set up and run the floris model
        self.fmodel.set(
            layout_x=inputs["x"],
            layout_y=inputs["y"],
            wind_data=self.time_series,
            yaw_angles=np.array([inputs["yaw"]]),
        )
        self.fmodel.set_operation_model("peak-shaving")

        self.fmodel.run()

        # dump the floris case
        self.fmodel.core.to_file(os.path.join(self.dir_floris, "batch.yaml"))

        outputs["power_farm"] = self.fmodel.get_farm_power()
        # outputs["powers_turbines"] = self.fmodel.get_turbine_powers()
        # outputs["Ct_turbines"] = self.fmodel.get_turbine_thrust_coefficients()
