import numpy as np
import jax.numpy as jnp
from jax import jit, jacobian
from ard.utils import distance_point_to_lineseg_nd, smooth_min, distance_lineseg_to_lineseg_nd
import openmdao.api as om

class MooringConstraint(om.ExplicitComponent):
    """
    A class to reduce complex mooring constraints into a simple violation lengthscale

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from `FarmAeroTemplate`)
    wind_query : floris.wind_data.WindRose
        a WindQuery objects that specifies the wind conditions that are to be
        computed
    bathymetry_data : BathymetryData
        a BathymetryData object to specify the bathymetry mesh/sampling

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    x_anchors : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    y_anchors : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = int(self.modeling_options["farm"]["N_turbines"])
        self.N_anchors = int(self.modeling_options["platform"]["N_anchors"])
        self.N_distances = int((self.N_turbines-1)*self.N_turbines/2)
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

        # set up inputs and outputs for mooring system
        self.add_input(
            "x_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", jnp.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "x_anchors",
            jnp.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_anchors",
            jnp.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        # ADD ADDITIONAL (DESIGN VARIABLE) INPUTS HERE!!!!!

        self.add_output(
            "violation_distance",
            jnp.zeros(self.N_distances),
            units="km",
        )  # consolidated violation length
        # ADD ADDITIONAL (DESIGN VARIABLE) OUTPUTS HERE!!!!!

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        x_anchors = inputs["x_anchors"]
        y_anchors = inputs["y_anchors"]
        print(x_turbines)
        print(y_turbines)
        print(x_anchors)
        print(y_anchors)

        # TODO extend this to allow for 3d, which should just require a version of the mooring_constraint_xy function in 3d
        distances = mooring_constraint_xy(x_turbines, y_turbines, x_anchors, y_anchors)
        print(distances)

        # replace the below with the final values
        outputs["violation_distance"] = distances

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        x_anchors = inputs["x_anchors"]
        y_anchors = inputs["y_anchors"]

        jacobian = mooring_constraint_xy_jac(x_turbines, y_turbines, x_anchors, y_anchors)

        print(jacobian)

        partials["violation_distance", "x_turbines"] = jacobian[0]
        partials["violation_distance", "y_turbines"] = jacobian[1]
        partials["violation_distance", "x_anchors"] = jacobian[2]
        partials["violation_distance", "y_anchors"] = jacobian[3]

def mooring_constraint_xy(x_turbines: np.ndarray, y_turbines: np.ndarray, x_anchors: np.ndarray, y_anchors: np.ndarray):
    """Mooring constraint calculation in 2 dimensions

    Args:
        x_turbines (np.ndarray): array of turbine x positions
        y_turbines (np.ndarray): array of turbine y positions
        x_anchors (np.ndarray): array of anchor x positions
        y_anchors (np.ndarray): array of anchor y positions

    Returns:
        np.ndarray: 1D array of distances with length (n_turbines - 1)*n_turbines/2
    """

    # convert inputs
    mooring_points = convert_inputs_x_y_to_xy(x_turbines, y_turbines, x_anchors, y_anchors)
    # calculate minimum distances between each set of moorings
    distances = calc_mooring_distances(mooring_points)

    return distances

mooring_constraint_xy_jac = jacobian(mooring_constraint_xy, argnums=[0,1,2,3])

def calc_mooring_distances(mooring_points: np.ndarray) -> np.ndarray:
    """Calculate the minimum distances between each set of mooring lines

    Args:
        mooring_points (np.ndarray): array of mooring points of shape (n_turbines, n_anchors+1, n_dimensions) where n_dimensions may be 2 or 3

    Returns:
        np.ndarray: 1D array of distances with length (n_turbines - 1)*n_turbines/2
    """

    n_turbines = mooring_points.shape[0]
    n_distances = int((n_turbines - 1)*n_turbines/2)

    distances = jnp.zeros(n_distances)

    k = 0
    for i in range(0, mooring_points.shape[0]-1):
        for j in range(mooring_points.shape[0]):
            if i == j:
                continue
            distances = distances.at[k].set(distance_mooring_to_mooring(mooring_points[i], mooring_points[j]))
            k += 1

    return distances
# calc_mooring_distances = jit(calc_mooring_distances)

def convert_inputs_x_y_to_xy(x_turbines: np.ndarray, y_turbines: np.ndarray, x_anchors: np.ndarray, y_anchors: np.ndarray) -> np.ndarray:
    """Convert from inputs of x for turbines, y for turbines, x for anchors, and y for anchors to single array for mooring specification
    that is of shape (n_turbines, n_anchors+1, 2). for each set of points, the turbine position is given first followed by the anchor positions

    Args:
        x_turbines (np.ndarray): array of turbine x positions
        y_turbines (np.ndarray): array of turbine y positions
        x_anchors (np.ndarray): array of anchor x positions
        y_anchors (np.ndarray): array of anchor y positions

    Returns:
        np.ndarray: all input information combined into a single array of shape (n_turbines, n_anchors+1, 2)
    """

    n_turbines = len(x_turbines)
    n_anchors = x_anchors.shape[1]

    xy = jnp.zeros((n_turbines, n_anchors+1, 2))

    for i in jnp.arange(0, n_turbines):
        xy = xy.at[i, 0, 0].set(x_turbines[i])
        xy = xy.at[i, 0, 1].set(y_turbines[i])
        for j in jnp.arange(1, n_anchors+1):
            xy = xy.at[i, j, 0].set(x_anchors[i, j-1])
            xy = xy.at[i, j, 1].set(y_anchors[i, j-1])

    return xy
# convert_inputs_x_y_to_xy = jit(convert_inputs_x_y_to_xy)

def convert_inputs_x_y_z_to_xyz(x_turbines: np.ndarray, y_turbines: np.ndarray, z_turbines: np.ndarray, x_anchors: np.ndarray, y_anchors: np.ndarray, z_anchors: np.ndarray, ) -> np.ndarray:
    """Convert from inputs of x for turbines, y for turbines, z for turbines, x for anchors, y for anchors, and z for anchors to single array for mooring specification
    that is of shape (n_turbines, n_anchors+1, 3). for each set of points, the turbine position is given first followed by the anchor positions

    Args:
        x_turbines (np.ndarray): array of turbine x positions
        y_turbines (np.ndarray): array of turbine y positions
        z_turbines (np.ndarray): array of turbine z positions
        x_anchors (np.ndarray): array of anchor x positions
        y_anchors (np.ndarray): array of anchor y positions
        z_anchors (np.ndarray): array of anchor z positions

    Returns:
        np.ndarray: all input information combined into a single array of shape (n_turbines, n_anchors+1, 3)
    """
    n_turbines = len(x_turbines)
    n_anchors = x_anchors.shape[1]

    xyz = jnp.zeros((n_turbines, n_anchors+1, 3))

    for i in jnp.arange(0, n_turbines):
        xyz = xyz.at[i, 0, 0].set(x_turbines[i])
        xyz = xyz.at[i, 0, 1].set(y_turbines[i])
        xyz = xyz.at[i, 0, 2].set(z_turbines[i])
        for j in jnp.arange(1, n_anchors+1):
            xyz = xyz.at[i, j, 0].set(x_anchors[i, j-1])
            xyz = xyz.at[i, j, 1].set(y_anchors[i, j-1])
            xyz = xyz.at[i, j, 2].set(z_anchors[i, j-1])

    return xyz
# convert_inputs_x_y_z_to_xyz = jit(convert_inputs_x_y_z_to_xyz)

def distance_point_to_mooring(point: np.ndarray, P_mooring: np.ndarray) -> float:
    """Find the distance from a point to a set of mooring lines for a single floating wind turbine.
        While arguments may be given in either 2d ([x,y]) or 3d ([x,y,z]), the point of interest 
        and the mooring line points must all be given with the same number of dimensions.

    Args:
        point (np.ndarray): Point of interest in 2d ([x,y]) or 3d ([x,y,z]).
        P_mooring (np.ndarray): The set of points defining the mooring line layout. The first point should
                                be the center, the rest of the points define the anchor points. Points may 
                                be given in 2d ([x,y]) or 3d ([x,y,z]).

    Returns:
        float: The shortest distance from the point of interest to the set of mooring lines.
    """

    p_center = P_mooring[0]
    distance_moorings = jnp.array([
        distance_point_to_lineseg_nd(point, 
                                     jnp.array(p_center), 
                                     jnp.array(p_anchor)) for p_anchor in P_mooring[1:]
                                ])
    
    return smooth_min(distance_moorings)
distance_point_to_mooring = jit(distance_point_to_mooring)

def distance_mooring_to_mooring(P_mooring_A: np.ndarray, P_mooring_B: np.ndarray) -> float:
    """Calculate the distance from one mooring to another. Moorings are defined with center point first
        followed by anchor points in no specific order.

    Args:
        P_mooring_A (np.ndarray): ndarray of points of mooring A of shape (npoints, nd) (e.g. (4, (x, y, z))).
            Center point must come first.
        P_mooring_B (np.ndarray): ndarray of points of mooring B of shape (npoints, nd) (e.g. (4, (x, y, z))).
            Center point must come first.

    Returns:
        float: shortest distance between the two sets of moorings
    """

    p_center_A = P_mooring_A[0]
    p_center_B = P_mooring_B[0]
    distance_moorings_b = jnp.array([[distance_lineseg_to_lineseg_nd(
        p_center_A,
        point_anchor_A,
        p_center_B,
        point_anchor_B
        ) for point_anchor_B in P_mooring_B[1:]] for point_anchor_A in P_mooring_A[1:]])

    return smooth_min(jnp.array([smooth_min(d) for d in distance_moorings_b]))
distance_mooring_to_mooring = jit(distance_mooring_to_mooring)