import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from ard.utils import distance_point_to_lineseg_nd, smooth_min, distance_lineseg_to_lineseg_nd
import openmdao.api as om
import jax
class MooringConstraint(om.ExplicitComponent):
    """
    A class to reduce complex mooring constraints into a simple violation lengthscale

    TODO

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
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_anchors = self.modeling_options["platform"]["N_anchors"]
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
            0.0,
            units="km",
        )  # consolidated violation length
        # ADD ADDITIONAL (DESIGN VARIABLE) OUTPUTS HERE!!!!!

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        x_anchors = inputs["x_anchors"]
        y_anchors = inputs["y_anchors"]

        ########################################################################
        #
        # this is where the magic will happen!
        #
        # here, the implementor will take the variables above and map them into
        # the machinery for computing a mooring design for each of the turbines.
        #
        ########################################################################

        raise NotImplementedError("This component is awaiting implementation!")

        # replace the below with the final values
        outputs["violation_distance"] = None

# TODO for 3d we just need an xyz version of this function
@jit
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

@jit
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