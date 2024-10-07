import numpy as np

import openmdao.api as om


class FarmAeroTemplate(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]

        # set up inputs and outputs for farm layout
        self.add_input("x", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y", np.zeros((self.N_turbines,)), units="m")
        self.add_input("yaw", np.zeros((self.N_turbines,)), units="deg")

    def compute(self, inputs, outputs):

        #############################################
        #                                           #
        # IMPLEMENT THE AERODYNAMICS COMPONENT HERE #
        #                                           #
        #############################################

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )


class BatchFarmPowerTemplate(FarmAeroTemplate):
    """
    template component for computing power using a farm aerodynamics model

    a farm power component, based on this template, will compute the power and
    thrust for a farm composed of a given rotor type.

    options
    -------
        - modeling_options : a modeling options dictionary to specify modeling
                specifications
        - wind_query : a WindQuery objects that specifies the wind conditions
                that are to be computed

    inputs
    ------
        - x : a 1D numpy array indicating the x-dimension locations of the
                turbines, with length N_turbines
        - y : a 1D numpy array indicating the y-dimension locations of the
                turbines, with length N_turbines
        - yaw : a numpy array indicating the yaw angle to drive each turbine to
                with respect to the ambient wind direction, with length
                N_turbines

    outputs
    -------
        - power_farm : a numpy array of the farm power for each of the wind
                conditions that have been queried
        - power_turbines : a numpy array of the farm power for each of the
                turbines in the farm across all of the conditions that have been
                queried (N_turbines, N_wind_conditions)
        - thrust_turbines : a numpy array of the wind turbine thrust for each of
                the turbines in the farm across all of the conditions that have
                been queried (N_turbines, N_wind_conditions)
    """

    def initialize(self):
        super().initialize()

        # farm power wind conditions query (not necessarily a full wind rose)
        self.options.declare("wind_query")

    def setup(self):
        super().setup()

        # unpack wind query object
        self.wind_query = self.options["wind_query"]
        self.directions_wind = self.options["wind_query"].get_directions()
        self.speeds_wind = self.options["wind_query"].get_speeds()
        if self.options["wind_query"].get_TIs() is None:
            self.options["wind_query"].set_TI_using_IEC_method()
        self.TIs_wind = self.options["wind_query"].get_TIs()
        self.N_wind_conditions = self.options["wind_query"].N_conditions

        # add the outputs we want for a batched power analysis:
        #   - farm and turbine powers
        #   - turbine thrusts
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

    def setup_partials(self):
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        super().compute(inputs, outputs)

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )

        # the following should be set
        outputs["power_farm"] = np.zeros((self.N_wind_conditions,))
        outputs["power_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))


class FarmAEPTemplate(FarmAeroTemplate):
    """
    template component for computing power using a farm aerodynamics model

    a farm power component, based on this template, will compute the power and
    thrust for a farm composed of a given rotor type.

    options
    -------
        - modeling_options : a modeling options dictionary to specify modeling
                specifications
        - wind_rose : a FLORIS WindRose object that fully specifies the wind
                conditions on which a farm is to be evaluated

    inputs
    ------
        - x : a 1D numpy array indicating the x-dimension locations of the
                turbines, with length N_turbines
        - y : a 1D numpy array indicating the y-dimension locations of the
                turbines, with length N_turbines
        - yaw : a numpy array indicating the yaw angle to drive each turbine to
                with respect to the ambient wind direction, with length
                N_turbines

    outputs
    -------
        - AEP_farm : a float giving the AEP of the farm
        - power_turbines : a numpy array of the farm power for each of the
                turbines in the farm across all of the conditions that have been
                queried on the wind rose (N_turbines, N_wind_conditions)
        - thrust_turbines : a numpy array of the wind turbine thrust for each of
                the turbines in the farm across all of the conditions that have
                been queried on the wind rose (N_turbines, N_wind_conditions)
    """

    def initialize(self):
        super().initialize()

        # wind conditions for AEP analysis are a FLORIS WindRose
        self.options.declare("wind_rose")  # FLORIS WindRose object

    def setup(self):
        super().setup()

        # unpack FLORIS wind data object
        self.wind_rose = self.options["wind_rose"]
        self.directions_wind, self.speeds_wind, self.TIs_wind, self.pmf_wind, _, _ = (
            self.wind_rose.unpack()
        )
        self.N_wind_conditions = len(self.pmf_wind)

        # add the outputs we want for an AEP analysis:
        #   - AEP estimate
        #   - farm and turbine powers
        #   - turbine thrusts
        self.add_output(
            "AEP_farm",
            0.0,
            units="W*h",
        )
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

        #############################################
        #                                           #
        # IMPLEMENT THE AERODYNAMICS COMPONENT HERE #
        #                                           #
        #############################################

        raise NotImplementedError(
            "This is an abstract class for a derived class to implement!"
        )

        # the following should be set
        outputs["AEP_farm"] = 0.0
        outputs["power_farm"] = np.zeros((self.N_wind_conditions,))
        outputs["power_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
        outputs["thrust_turbines"] = np.zeros((self.N_turbines, self.N_wind_conditions))
