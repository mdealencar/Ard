import numpy as np

import openmdao.api as om


class BatchFarmPowerTemplate(om.ExplicitComponent):
    """
    template component for computing power using a farm aerodynamics model


    """

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("wind_query")

    def setup(self):
        # load modeling options
        modeling_options = self.modeling_options = self.options["modeling_options"]
        self.N_turbines = modeling_options["farm"]["N_turbines"]

        # unpack wind query object
        self.wind_query = self.options["wind_query"]
        self.directions_wind = self.options["wind_query"].get_directions()
        self.speeds_wind = self.options["wind_query"].get_speeds()
        self.TIs_wind = (
            self.options["wind_query"].get_TIs()
            if self.options["wind_query"].get_TIs()
            else 0.06 * np.ones_like(self.directions_wind)
        )
        self.N_wind_conditions = self.options["wind_query"].N_conditions

        # set up inputs and outputs
        self.add_input("x", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y", np.zeros((self.N_turbines,)), units="m")
        self.add_input("yaw", np.zeros((self.N_turbines,)), units="deg")
        # ... more inputs can be added here

        self.add_output(
            "power_farm",
            np.zeros((self.N_wind_conditions,)),
            units="W",
        )
        self.add_output(
            "power_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="W",
        )
        self.add_output(
            "thrust_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="N",
        )
        # ... more outputs can be added here

    def setup_partials(self):
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )

        # the following should be set
        outputs["power_farm"] = 0.0
        outputs["power_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))


class FarmAEPTemplateComponent(om.ExplicitComponent):
    """
    template component for computing AEP using a farm aerodynamics model


    """

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("wind_rose")  # FLORIS WindRose object

    def setup(self):
        # load modeling options
        modeling_options = self.modeling_options = self.options["modeling_options"]
        self.N_turbines = modeling_options["farm"]["N_turbines"]

        # unpack FLORIS wind data object
        self.wind_rose = self.options["wind_rose"]
        self.directions_wind, self.speeds_wind, self.TIs_wind, self.pmf_wind, _, _ = (
            self.wind_rose.unpack()
        )
        self.N_wind_conditions = len(self.pmf_wind)

        # set up inputs and outputs
        self.add_input("x", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y", np.zeros((self.N_turbines,)), units="m")
        self.add_input("yaw", np.zeros((self.N_turbines,)), units="deg")
        # ... more inputs can be added here

        self.add_output(
            "AEP_farm",
            0.0,
            units="W*h",
        )
        self.add_output(
            "power_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="W",
        )
        self.add_output(
            "thrust_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="N",
        )
        # ... more outputs can be added here

    def setup_partials(self):
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )

        # the following should be set
        outputs["power_farm"] = 0.0
        outputs["power_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
