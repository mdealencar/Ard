import numpy as np
import openmdao.api as om

import shapely.geometry as sg

import windard.layout.templates as templates


class FullFarmLanduse(templates.LanduseTemplate):
    """
    a class to compute the land use of a full layout optimization
    """

    def setup(self):
        super().setup()

        # add the full layout inputs
        self.add_input(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in x-direction",
        )
        self.add_input(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in y-direction",
        )

    def compute(self, inputs, outputs):

        # extract the points from the inputs
        points = list(
            zip(
                list(inputs["x_turbines"]),
                list(inputs["y_turbines"]),
            )
        )

        # create a multi-point object
        mp = sg.MultiPoint(points)

        # area tight is equal to the convex hull area for the points in sq. km.
        outputs["area_tight"] = mp.convex_hull.area / 1000**2
