import numpy as np

import windard.layout.templates as templates


class GridFarmLayout(templates.LayoutTemplate):
    """
     a class to take a parameterized, structured grid farm and output an actual
     grid for the farm

                                      |-------| <- streamwise spacing
      orient.         x ----- x ----- x ----- x ----- x -
       angle         /       /       /       /       /  | <- spanwise spacing
        |           x ----- x ----- x ----- x ----- x   -      (perpendicular
        v          /       /       /       /       /            w.r.t. primary)
       -------    x ----- x ----- x ----- x ----- x    ----- primary vector
            '    /       /       /       /       /             (rotated from
        '       x ----- x ----- x ----- x ----- x               north CW by
    '          /       /       /       /       /                orientation
     NORTH    x ----- x ----- x ----- x ----- x                 angle)
                     /|
                    / |
                   /  | <- skew angle
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        super().setup()

        # add four-parameter grid farm layout DVs
        self.add_input("spacing_primary", 7.0)
        self.add_input("spacing_secondary", 7.0)
        self.add_input("angle_orientation", 0.0, units="deg")
        self.add_input("angle_skew", 0.0, units="deg")

    def compute(self, inputs, outputs):

        D_rotor = self.modeling_options["turbine"]["geometry"]["diameter_rotor"]
        lengthscale_spacing_streamwise = inputs["spacing_primary"] * D_rotor
        lengthscale_spacing_spanwise = inputs["spacing_secondary"] * D_rotor

        N_square = int(np.sqrt(self.N_turbines))  # floors

        count_y, count_x = np.meshgrid(
            np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2)),
            np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2)),
        )

        if self.N_turbines == N_square**2:
            pass
        elif self.N_turbines <= N_square * (N_square + 1):
            # grid farm is a little bit above the last square... add a trailing
            # row.
            count_x = np.vstack([count_x, ((N_square + 1) / 2) * np.ones((N_square,))])
            count_y = np.vstack(
                [count_y, np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2))]
            )
            count_x = count_x.flatten()
            count_y = count_y.flatten()
        else:
            # grid farm is nearly the next square... oversize and cut the last
            count_y, count_x = np.meshgrid(
                np.arange(-((N_square) / 2), ((N_square + 2) / 2)),
                np.arange(-((N_square) / 2), ((N_square + 2) / 2)),
            )
        count_x = count_x.flatten()[: self.N_turbines]
        count_y = count_y.flatten()[: self.N_turbines]

        angle_skew = -np.radians(inputs["angle_skew"])
        xf_positions = (
            count_x * lengthscale_spacing_streamwise
            + count_y * lengthscale_spacing_spanwise * np.tan(angle_skew)
        )
        yf_positions = count_y * lengthscale_spacing_spanwise

        angle_orientation = np.radians(inputs["angle_orientation"])
        Amtx = np.array(
            [
                [np.sin(angle_orientation), np.cos(angle_orientation)],
                [np.cos(angle_orientation), -np.sin(angle_orientation)],
            ]
        ).squeeze()
        xyp = Amtx @ np.vstack([xf_positions, yf_positions])

        outputs["x_turbines"] = xyp[0, :].tolist()
        outputs["y_turbines"] = xyp[1, :].tolist()

        outputs["spacing_effective_secondary"] = inputs["spacing_primary"]
        outputs["spacing_effective_secondary"] = np.sqrt(
            inputs["spacing_secondary"] ** 2.0 / np.cos(angle_skew) ** 2.0
        )


class GridFarmLanduse(templates.LanduseTemplate):
    """
    a class that can compute the land-use area of the above parametrized,
    structured grid farm above and output the land use for the farm
    """

    def setup(self):
        super().setup()

        # add grid farm-specific inputs
        self.add_input("spacing_primary", 7.0)
        self.add_input("spacing_secondary", 7.0)
        self.add_input("angle_orientation", 0.0, units="deg")
        self.add_input("angle_skew", 0.0, units="deg")

        self.add_output(
            "area_aligned_parcel",
            0.0,
            units="km**2",
            desc="area of the tightest rectangle around the farm (plus layback) that is aligned with the orientation vector",
        )
        self.add_output(
            "area_compass_parcel",
            0.0,
            units="km**2",
            desc="area of the tightest rectangular and compass-aligned land parcel that will fit the farm (plus layback)",
        )

    def compute(self, inputs, outputs):

        D_rotor = self.modeling_options["turbine"]["geometry"]["diameter_rotor"]
        lengthscale_spacing_streamwise = inputs["spacing_primary"] * D_rotor
        lengthscale_spacing_spanwise = inputs["spacing_secondary"] * D_rotor
        lengthscale_layback = inputs["distance_layback_diameters"] * D_rotor

        N_square = int(np.sqrt(self.N_turbines))  # floors

        min_count_xf = -(N_square - 1) / 2
        min_count_yf = -(N_square - 1) / 2
        max_count_xf = (N_square - 1) / 2
        max_count_yf = (N_square - 1) / 2

        if self.N_turbines == N_square**2:
            pass
        elif self.N_turbines <= N_square * (N_square + 1):
            max_count_xf = (N_square + 1) / 2
        else:
            min_count_xf = -(N_square) / 2
            min_count_yf = -(N_square) / 2
            max_count_xf = (N_square) / 2
            max_count_yf = (N_square) / 2

        # the side lengths of a parallelopiped oriented with the farm that encloses the farm with layback
        length_farm_xf = (max_count_xf - min_count_xf) * lengthscale_spacing_streamwise
        length_farm_yf = (max_count_yf - min_count_yf) * lengthscale_spacing_spanwise

        # the area of a parallelopiped oriented with the farm that encloses the farm with layback
        area_parallelopiped = (length_farm_xf + 2 * lengthscale_layback) * (
            length_farm_yf + 2 * lengthscale_layback
        )

        # the side lengths of a square oriented with the farm that encloses the farm with layback
        angle_skew = np.radians(inputs["angle_skew"])
        length_enclosing_farm_xf = (
            max_count_xf * lengthscale_spacing_streamwise
            + max_count_yf * lengthscale_spacing_spanwise * np.abs(np.tan(angle_skew))
        ) - (
            min_count_xf * lengthscale_spacing_streamwise
            + min_count_yf * lengthscale_spacing_spanwise * np.abs(np.tan(angle_skew))
        )
        length_enclosing_farm_yf = length_farm_yf

        # the area of a square oriented with the farm that encloses the farm with layback
        area_enclosingsquare_farmoriented = (
            length_enclosing_farm_xf + 2 * lengthscale_layback
        ) * (length_enclosing_farm_yf + 2 * lengthscale_layback)

        # the side lengths of a square oriented with the compass rose that encloses the farm with layback
        angle_orientation = np.radians(inputs["angle_orientation"])
        length_enclosing_farm_x = (
            np.cos(angle_orientation) * length_enclosing_farm_xf
            + np.abs(np.sin(angle_orientation)) * length_enclosing_farm_yf
        )
        length_enclosing_farm_y = (
            np.abs(np.sin(angle_orientation)) * length_enclosing_farm_xf
            + np.cos(angle_orientation) * length_enclosing_farm_yf
        )
        area_enclosingsquare_compass = (
            length_enclosing_farm_x + 2 * lengthscale_layback
        ) * (length_enclosing_farm_y + 2 * lengthscale_layback)

        outputs["area_tight"] = area_parallelopiped / (1e3) ** 2
        outputs["area_aligned_parcel"] = area_enclosingsquare_farmoriented / (1e3) ** 2
        outputs["area_compass_parcel"] = area_enclosingsquare_compass / (1e3) ** 2
