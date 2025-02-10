import numpy as np
import scipy.spatial

import ard.layout.templates as templates
import ard.layout.fullfarm as fullfarm


phi_golden = (1 + np.sqrt(5)) / 2  # golden ratio


def sunflower(
    n: float,
    alpha: float = 0,  # proportion of points that should end on boundary
    n_b: float = None,  # for overriding with the number of boundary elements
    geodesic=False,  # use geodesic step function
):
    """
    generate a sunflower seed packing pattern

    adapted from a stackoverflow post:
        https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle#28572551

    in turn from the wolfram demonstrations page:
        Joost de Jong (2013),
        "Sunflower Seed Arrangements"
        Wolfram Demonstrations Project.
        demonstrations.wolfram.com/SunflowerSeedArrangements/.

    appears to originate from: doi:10.1016/0025-5564(79)90080-4
    """

    def radius(k: int, n: int, b: int):
        """
        radius at which a seed should live
        b sets the number of boundary points
        remainder of n points have sequence-baased location
        """

        if k > n - b:
            return 1.0
        else:
            return np.sqrt(k - 0.5) / np.sqrt(n - (b + 1) / 2)

    points = []  # initialize a set of points
    # each next angle should step by a certain amount
    angle_stride = 2 * np.pi * phi_golden if geodesic else 2 * np.pi / phi_golden**2
    b = (
        n_b if n_b is not None else round(alpha * np.sqrt(n))
    )  # number of boundary points
    for k in range(1, n + 1):
        r = radius(k, n, b)  # get radius
        theta = k * angle_stride  # get angle
        points.append((r * np.cos(theta), r * np.sin(theta)))
    return points


class SunflowerFarmLayout(templates.LayoutTemplate):
    """
    A sunflower-inspired structured layout algorithm

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from
        `templates.LayoutTemplate`)
    N_turbines : int
        the number of turbines that should be in the farm layout (inherited from
        `templates.LayoutTemplate`)

    Inputs
    ------
    alpha : float
        a parameter to control the number of boundary (v. interior) turbines

    Outputs
    -------
    x_turbines : np.ndarray
        a 1-D numpy array that represents that x (i.e. Easting) coordinate of
        the location of each of the turbines in the farm in meters (inherited
        from `templates.LayoutTemplate`)
    y_turbines : np.ndarray
        a 1-D numpy array that represents that y (i.e. Northing) coordinate of
        the location of each of the turbines in the farm in meters (inherited
        from `templates.LayoutTemplate`)
    spacing_effective_primary : float
        a measure of the spacing on a primary axis of a rectangular farm that
        would be equivalent to this one for the purposes of computing BOS costs
        measured in rotor diameters (inherited from `templates.LayoutTemplate`)
    spacing_effective_secondary : float
        a measure of the spacing on a secondary axis of a rectangular farm that
        would be equivalent to this one for the purposes of computing BOS costs
        measured in rotor diameters (inherited from `templates.LayoutTemplate`)
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

    def setup(self):
        """Setup of OM component."""
        super().setup()

        # add parameters for sunflower farm DVs
        # self.add_input("alpha", 0.0, desc="boundary point control param.")
        self.add_input("spacing_target", 0.0, desc="target spacing in rotor diameters")

    def setup_partials(self):
        """Derivative setup for OM component."""

        # default complex step for the layout tools, since they're often algebraic
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        """Computation for the OM component."""

        # get the desired mean nearest-neighbor distance
        D_rotor = self.modeling_options["turbine"]["geometry"][
            "diameter_rotor"
        ]  # get rotor diameter
        spacing_target = D_rotor * inputs["spacing_target"]

        # generate the points
        points = np.array(sunflower(self.N_turbines, geodesic=True))
        dist_mtx = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(points)
        )
        np.fill_diagonal(dist_mtx, np.inf)  # self-distance not meaningful, remove
        d_mean_NN = np.mean(np.min(dist_mtx, axis=0))

        points *= spacing_target / d_mean_NN

        # assert np.isclose(
        #     np.mean(
        #         np.min(
        #             np.fill_diagonal(
        #                 scipy.spatial.distance.squareform(
        #                     scipy.spatial.distance.pdist(points)
        #                 ),
        #                 np.inf,
        #             ),
        #             axis=0,
        #         )
        #     ),
        #     spacing_target,
        # )

        print("DEBUG!!!!!", points)

        outputs["x_turbines"] = points[:,0].tolist()
        outputs["y_turbines"] = points[:,1].tolist()

        outputs["spacing_effective_primary"] = spacing_target  # ???
        outputs["spacing_effective_secondary"] = spacing_target  # ???


class SunflowerFarmLanduse(fullfarm.FullFarmLanduse):
    pass


# class SunflowerFarmLanduse(templates.LanduseTemplate):
#     """
#     Landuse class for four-parameter parallelepiped grid farm layout.
#
#     This is a class to compute the landuse area of the parameterized, structured
#     grid farm defined in `SunflowerFarmLayout`.
#
#     Options
#     -------
#     modeling_options : dict
#         a modeling options dictionary (inherited from
#         `templates.LayoutTemplate`)
#     N_turbines : int
#         the number of turbines that should be in the farm layout (inherited from
#         `templates.LayoutTemplate`)
#
#     Inputs
#     ------
#     distance_layback_diameters : float
#         the number of diameters of layback desired for the landuse calculation
#         (inherited from `templates.LayoutTemplate`)
#     angle_orientation : float
#         orientation in degrees clockwise with respect to North of the primary
#         axis of the wind farm layout
#     spacing_primary : float
#         spacing of turbine rows along the primary axis (rotated by
#         `angle_orientation`) in nondimensional rotor diameters
#     spacing_secondary : float
#         spacing of turbine columns along the perpendicular to the primary axis
#         (rotated by 90° with respect to the primary axis) in nondimensional
#         rotor diameters
#     angle_skew : float
#         clockwise skew angle of turbine rows w.r.t. beyond the 90° clockwise
#         perpendicular to the primary axis
#
#     Outputs
#     -------
#     area_tight : float
#         the area in square kilometers that the farm occupies based on the
#         circumscribing geometry with a specified (default zero) layback buffer
#         (inherited from `templates.LayoutTemplate`)
#     area_aligned_parcel : float
#         the area in square kilometers that the farm occupies based on the
#         circumscribing rectangle that is aligned with the primary axis of the
#         wind farm plus a specified (default zero) layback buffer
#     area_compass_parcel : float
#         the area in square kilometers that the farm occupies based on the
#         circumscribing rectangle that is aligned with the compass rose plus a
#         specified (default zero) layback buffer
#     """
#
#     def setup(self):
#         """Setup of OM component."""
#
#         super().setup()
#
#         # add grid farm-specific inputs
#         self.add_input("spacing_primary", 7.0)
#         self.add_input("spacing_secondary", 7.0)
#         self.add_input("angle_orientation", 0.0, units="deg")
#         self.add_input("angle_skew", 0.0, units="deg")
#
#         self.add_output(
#             "area_aligned_parcel",
#             0.0,
#             units="km**2",
#             desc="area of the tightest rectangle around the farm (plus layback) that is aligned with the orientation vector",
#         )
#         self.add_output(
#             "area_compass_parcel",
#             0.0,
#             units="km**2",
#             desc="area of the tightest rectangular and compass-aligned land parcel that will fit the farm (plus layback)",
#         )
#
#     def setup_partials(self):
#         """Derivative setup for OM component."""
#
#         # default complex step for the layout tools, since they're often algebraic
#         self.declare_partials("*", "*", method="cs")
#
#     def compute(self, inputs, outputs):
#         """Computation for the OM component."""
#
#         D_rotor = self.modeling_options["turbine"]["geometry"]["diameter_rotor"]
#         lengthscale_spacing_streamwise = inputs["spacing_primary"] * D_rotor
#         lengthscale_spacing_spanwise = inputs["spacing_secondary"] * D_rotor
#         lengthscale_layback = inputs["distance_layback_diameters"] * D_rotor
#
#         N_square = int(np.sqrt(self.N_turbines))  # floors
#
#         min_count_xf = -(N_square - 1) / 2
#         min_count_yf = -(N_square - 1) / 2
#         max_count_xf = (N_square - 1) / 2
#         max_count_yf = (N_square - 1) / 2
#
#         if self.N_turbines == N_square**2:
#             pass
#         elif self.N_turbines <= N_square * (N_square + 1):
#             max_count_xf = (N_square + 1) / 2
#         else:
#             min_count_xf = -(N_square) / 2
#             min_count_yf = -(N_square) / 2
#             max_count_xf = (N_square) / 2
#             max_count_yf = (N_square) / 2
#
#         # the side lengths of a parallelopiped oriented with the farm that encloses the farm with layback
#         length_farm_xf = (max_count_xf - min_count_xf) * lengthscale_spacing_spanwise
#         length_farm_yf = (max_count_yf - min_count_yf) * lengthscale_spacing_streamwise
#
#         # the area of a parallelopiped oriented with the farm that encloses the farm with layback
#         area_parallelopiped = (length_farm_xf + 2 * lengthscale_layback) * (
#             length_farm_yf + 2 * lengthscale_layback
#         )
#
#         # the side lengths of a square oriented with the farm that encloses the farm with layback
#         angle_skew = np.pi / 180.0 * inputs["angle_skew"]
#         length_enclosing_farm_xf = length_farm_xf
#         length_enclosing_farm_yf = (
#             max_count_yf * lengthscale_spacing_streamwise
#             + np.abs(max_count_xf)
#             * lengthscale_spacing_spanwise
#             * np.abs(np.tan(angle_skew))
#         ) - (
#             min_count_yf * lengthscale_spacing_streamwise
#             - np.abs(min_count_xf)
#             * lengthscale_spacing_spanwise
#             * np.abs(np.tan(angle_skew))
#         )
#
#         # the area of a square oriented with the farm that encloses the farm with layback
#         area_enclosingsquare_farmoriented = (
#             length_enclosing_farm_xf + 2 * lengthscale_layback
#         ) * (length_enclosing_farm_yf + 2 * lengthscale_layback)
#
#         # the side lengths of a square oriented with the compass rose that encloses the farm with layback
#         angle_orientation = np.pi / 180.0 * inputs["angle_orientation"]
#         A_x = (
#             +max_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
#             + max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
#             - (max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
#             * np.tan(angle_skew)
#         )
#         A_y = (
#             +max_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
#             - max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
#             - (max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
#             * np.tan(angle_skew)
#         )
#         B_x = (
#             +min_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
#             + max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
#             - (max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
#             * np.tan(angle_skew)
#         )
#         B_y = (
#             +min_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
#             - max_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
#             - (max_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
#             * np.tan(angle_skew)
#         )
#         C_x = (
#             min_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
#             + min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
#             - (min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
#             * np.tan(angle_skew)
#         )
#         C_y = (
#             min_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
#             - min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
#             - (min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
#             * np.tan(angle_skew)
#         )
#         D_x = (
#             -min_count_yf * lengthscale_spacing_streamwise * np.sin(angle_orientation)
#             + min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation)
#             - (min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation))
#             * np.tan(angle_skew)
#         )
#         D_y = (
#             -min_count_yf * lengthscale_spacing_streamwise * np.cos(angle_orientation)
#             - min_count_xf * lengthscale_spacing_spanwise * np.sin(angle_orientation)
#             - (min_count_xf * lengthscale_spacing_spanwise * np.cos(angle_orientation))
#             * np.tan(angle_skew)
#         )
#
#         length_enclosing_farm_x = np.max([A_x, B_x, C_x, D_x]) - np.min(
#             [A_x, B_x, C_x, D_x]
#         )
#         length_enclosing_farm_y = np.max([A_y, B_y, C_y, D_y]) - np.min(
#             [A_y, B_y, C_y, D_y]
#         )
#
#         area_enclosingsquare_compass = (
#             length_enclosing_farm_x + 2 * lengthscale_layback
#         ) * (length_enclosing_farm_y + 2 * lengthscale_layback)
#         outputs["area_tight"] = area_parallelopiped / (1e3) ** 2
#         outputs["area_aligned_parcel"] = area_enclosingsquare_farmoriented / (1e3) ** 2
#         outputs["area_compass_parcel"] = area_enclosingsquare_compass / (1e3) ** 2
