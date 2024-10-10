import numpy as np

import openmdao.api as om


class LayoutTemplate(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        # load modeling options
        modeling_options = self.modeling_options = self.options["modeling_options"]
        self.N_turbines = modeling_options["farm"]["N_turbines"]

        # add outputs that are universal
        self.add_output(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in x-direction",
        )
        self.add_output(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in y-direction",
        )
        self.add_output(
            "spacing_effective_primary",
            0.0,
            desc="effective spacing in x-dimension for BOS calculation",
        )
        self.add_output(
            "spacing_effective_secondary",
            0.0,
            desc="effective spacing in y-dimension for BOS calculation",
        )

    def setup_partials(self):
        # default complex step for the layout tools, since they're often algebraic
        self.declare_partials("*", "*", method="cs")

    def compute(self):
        raise NotImplementedError(
            "This is an abstract class for a derived class to implement"
        )
