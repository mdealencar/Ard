import copy
from os import PathLike
from pathlib import Path
import yaml
import jax.numpy as jnp

import numpy as np

from wisdem.inputs.validation import load_yaml

# def distance_point_to_lineseg(point_x: float, point_y: float, line_a_x: float, line_a_y: float, line_b_x: float, line_b_y: float, k_logistic: float=100.0) -> float:
#     """Find the distance between a point and a finite line segment

#     Args:
#         point_x (float): x coordinate of the point of interest
#         point_y (float): y coordinate of the point of interest
#         line_a_x (float): x coordinate of point A of the line AB
#         line_a_y (float): y coordinate of point A of the line AB
#         line_b_x (float): x coordinate of point B of the line AB
#         line_b_y (float): y coordinate of point B of the line AB
#         k_logistic (float, optional): _description_. Defaults to 100.0.

#     Returns:
#         float: distance from the point of interest to line AB
#     """

#     # length of the line segment squared
#     L2 = (line_b_x - line_a_x) ** 2 + (line_b_y - line_a_y) ** 2

#     t = ((point_x - line_a_x) * (line_b_x - line_a_x) + (point_y - line_a_y) * (line_b_y - line_a_y)) / L2
#     x_P = line_a_x + t * (line_b_x - line_a_x)
#     y_P = line_a_y + t * (line_b_y - line_a_y)

#     d_CA = np.sqrt((point_x - line_a_x) ** 2 + (point_y - line_a_y) ** 2)
#     d_CB = np.sqrt((point_x - line_b_x) ** 2 + (point_y - line_b_y) ** 2)
#     d_CP = np.sqrt((point_x - x_P) ** 2 + (point_y - y_P) ** 2)

#     soft_filter_t0 = (
#         1.0 / (1.0 + np.exp(-k_logistic * (t - 0.0)))
#         if (-k_logistic * t < 100)
#         else 0.0
#     )
#     soft_filter_t1 = (
#         1.0 / (1.0 + np.exp(-k_logistic * (1.0 - t))) if (k_logistic * t < 100) else 0.0
#     )
#     return (
#         d_CP * (soft_filter_t0) * (soft_filter_t1)
#         + d_CA * (1.0 - soft_filter_t0)
#         + d_CB * (1.0 - soft_filter_t1)
#     )

def _distance_lineseg_to_lineseg_coplanar_no_intersect_nd(line_a_start: np.ndarray, line_a_end: np.ndarray, line_b_start: np.ndarray, line_b_end: np.ndarray) -> float:
    """Returns the distance between two finite line segments assuming the segments are coplanar and do not intersect. It is up to the user to check the intersect condition

    Args:
        line_a_start (np.ndarray): start point of line a
        line_a_end (np.ndarray): end point of line a
        line_b_start (np.ndarray): start point of line b
        line_b_end (np.ndarray): end point of line b

    Returns:
        distance (float): the distance between the lines
    """
    # check that inputs are coplanar
    
    if jnp.linalg.vecdot((line_b_start - line_a_start), (jnp.linalg.cross(line_a_end, line_b_end))) != 0.0:
        raise(ValueError("The two lines provided must be coplanar"))
    
    # get distance between all pairs of end points
    a_start_to_b = distance_point_to_lineseg_nd(line_a_start, line_b_start, line_b_end)
    a_end_to_b = distance_point_to_lineseg_nd(line_a_end, line_b_start, line_b_end)
    b_start_to_a = distance_point_to_lineseg_nd(line_b_start, line_a_start, line_a_end)
    b_end_to_a = distance_point_to_lineseg_nd(line_b_end, line_a_start, line_a_end)

    distances = jnp.array([a_start_to_b, a_end_to_b, b_start_to_a, b_end_to_a])

    # import pdb; pdb.set_trace()

    distance = smooth_min(distances)

    return distance

def distance_lineseg_to_lineseg_nd(line_a_start: np.ndarray, line_a_end: np.ndarray, line_b_start: np.ndarray, line_b_end: np.ndarray) -> float:

    # [1] Numerical Recipes: The Art of Scientific Computing by Press, et al. 3rd edition
    line_a_vector = line_a_end - line_a_start 
    line_b_vector = line_b_end - line_b_start

    # check if line a is a point and get distance accordingly
    if jnp.all(line_a_vector == 0.0):
        distance = distance_point_to_lineseg_nd(line_a_start, line_b_start, line_b_end)

    # check if line b is a point and get distance accordingly
    elif jnp.all(line_b_vector == 0.0):
        distance = distance_point_to_lineseg_nd(line_b_start, line_a_start, line_a_end)
    
    # check if any points are shared and return the appropriate distance in a differentiable way
    elif jnp.all(line_a_start == line_b_start) or jnp.all(line_a_start == line_b_end) or jnp.all(line_a_end == line_b_start) or jnp.all(line_a_end == line_b_end):
        points_a = jnp.array([line_a_start, line_a_end])
        points_b = jnp.array([line_b_start, line_b_end])
        distance = np.nan
        for p_a in points_a:
            for p_b in points_b:
                if jnp.all(p_a == p_b):
                    # This will give the correct distance and approximate derivatives. If multiple points are the same, then this may lead to more incorrect derivatives
                    distance = sum((p_b - p_a)**2)
                    break

    # get distance between the line segments
    else:
        a = line_a_start
        v = line_a_end - line_a_start
        x = line_b_start
        u = line_b_end - line_b_start

        # find s and t (point along segment where the segments are closest to each other) using eq. 21.4.17 in [1]
        denominator = smooth_norm(jnp.cross(u, v))**2

        # denominator goes to zero when lines are parallel, so a different method must be used
        if denominator == 0.0:
            distance = _distance_lineseg_to_lineseg_coplanar_no_intersect_nd(line_a_start=line_a_start, line_a_end=line_a_end, line_b_start=line_b_start, line_b_end=line_b_end)
    
        else:
            s_numerator = jnp.linalg.det(jnp.array([a - x, u, jnp.linalg.cross(u, v)]).T)
            
            t_numerator = jnp.linalg.det(jnp.array([a - x, v, jnp.linalg.cross(u, v)]).T)

            s = s_numerator/denominator
            t = t_numerator/denominator

            # Get closest point along the lines 

            # if s > 1, use end point of line a by setting s to 1
            if s > 1:
                closest_point_line_a = line_a_end
            # if s < 0, use start point of line a by setting s to 0
            elif s < 0:
                closest_point_line_a = line_a_start
            # otherwise compute the closest point on line a using the parametric form of the line segment
            else:
                closest_point_line_a = a + s*v

            # if t > 1, use end point of line b by setting t to 1
            if t > 1:
                closest_point_line_b = line_b_end
            # if t < 1, use start point of line b by setting t to 0
            elif t < 0:
                closest_point_line_b = line_b_start
            # otherwise compute the closest point on line a using the parametric form of the line segment
            else:
                closest_point_line_b = x + t*u

            # the distance between the line segments is the distance between the closest points
            distance = smooth_norm(closest_point_line_b - closest_point_line_a)

    return distance

def distance_point_to_lineseg_nd(point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> float:
    """Find the distance from a point to a finite line segment in N-Dimensions

    Args:
        point (np.ndarray): point of interest [x,y,...]
        segment_start (np.ndarray): start point of line segment [x,y,...]
        segment_end (np.ndarray): end point of line segment [x,y,...]

    Returns:
        distance (float): shortest distance between the point and finite line segment
    """
    
    # get the vector of the line segment
    segment_vector = segment_end - segment_start

    # if the segment is a point, then get the distance to that point
    if jnp.all(segment_vector == 0):
        if np.all((segment_start - point) == 0.0):
            return 
        distance = smooth_norm(segment_start - point)

    else:
        # calculate the distance to the starting point
        start_to_point_vector = point - segment_start

        # calculate the unit vector projection of the start to point vector on the line segment
        projection = jnp.dot(start_to_point_vector, segment_vector) / jnp.dot(segment_vector, segment_vector)

        # if projection is outside the segment, then the unit projection will be negative and the start is the closest point
        if projection < 0:
            closest_point = segment_start
        # if the project is greater than 1, then the end point is the closest point
        elif projection > 1:
            closest_point = segment_end
        # otherwise, find the point of the intersection of the line and a perpendicular intersect through the point
        else:
            closest_point = segment_start + projection*segment_vector

        # the distance from the point to the line is the distance from the point to the closest point on the line
        distance = smooth_norm(point - closest_point)

    return distance

def smooth_max(x:jnp.ndarray, s:float=1000.0) -> float:
    """Non-overflowing version of Smooth Max function (see ref 3 and 4 below). 
    Calculates the smoothmax (a.k.a. softmax or LogSumExponential) of the elements in x.

    Based on implementation in BYU FLOW Lab's FLOWFarm software at
    (1) https://github.com/byuflowlab/FLOWFarm.jl/tree/master
    which is based on John D. Cook's writings at
    (2) https://www.johndcook.com/blog/2010/01/13/soft-maximum/
    and
    (3) https://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
    And based on article in FeedlyBlog
    (4) https://blog.feedly.com/tricks-of-the-trade-logsumexp/

    Args:
        x (list): list of values to be compared
        s (float, optional): alpha for smooth max function. Defaults to 100.0. 
            Larger values of `s` lead to more accurate results, but reduce the smoothness 
            of the output values.

    Returns:
        float: the smooth max of the provided `x` list
    """

    # get the maximum value and the index of maximum value
    max_ind = jnp.argmax(x)
    max_val = x[max_ind]
    
    # get the indices of x
    indices = jnp.arange(0, len(x), dtype=int)

    # LogSumExp with smoothing factor s
    exponential = jnp.exp(s*(jnp.array(x)[indices != max_ind] - max_val))
    r = (jnp.log(1.0 + jnp.sum(exponential)) + s*max_val)/s

    return r

def smooth_min(x:np.ndarray, s:float=1000.0) -> float:
    """ Finds smooth min using the `smooth_max` function

    Args:
        x (list): list of values to be compared
        s (float, optional): alpha for smooth min function. Defaults to 100.0. 
            Larger values of `s` lead to more accurate results, but reduce the smoothness 
            of the output values.

    Returns:
        float: the smooth min of the provided `x` list
    """

    return -smooth_max(x=-x, s=s)

def smooth_norm(vec: np.ndarray, buf: float=1E-12) -> float:
    """Smooth version of the Frobenius, or 2, norm. This version is nearly equivalent to the 2-norm with the 
    maximum absolute error corresponding to the order of the buffer value. The maximum error in the gradient is near unity, but 
    the error is generally about twice the error in the absolute fidelity. The key benefit of the smooth norm is 
    that it is differentiable near zero, while the standard norm is undefined.

    Args:
        vec (np.ndarray): input vector to be normed
        buf (float, optional): buffer value included in the sum of squares part of the norm. Defaults to 1E-6.

    Returns:
        (float): normed result
    """
    return jnp.sqrt(buf**2 + jnp.sum(vec**2)) 

def load_turbine_spec(
    filename_turbine_spec: PathLike,
):
    """
    Utility to load turbine spec. files, & transform relative to absolute links.

    Parameters
    ----------
    filename_turbine_spec : Pathlike
        filename of a turbine specification to load

    Returns
    -------
    dict
        a turbine specification dictionary
    """

    filename_turbine_spec = Path(filename_turbine_spec)
    dir_turbine_spec = filename_turbine_spec.parent
    turbine_spec = load_yaml(filename_turbine_spec)
    filename_powercurve = (
        dir_turbine_spec / turbine_spec["performance_data_ccblade"]["power_thrust_csv"]
    )
    turbine_spec["performance_data_ccblade"]["power_thrust_csv"] = filename_powercurve

    return turbine_spec


def create_FLORIS_turbine(
    input_turbine_spec: dict | PathLike,
    filename_turbine_FLORIS: PathLike = None,
) -> dict:
    """
    Create a FLORIS turbine from a generic Ard turbine specification.

    Parameters
    ----------
    input_turbine_spec : dict | PathLike
        a turbine specification from which to extract a FLORIS turbine
    filename_turbine_FLORIS : PathLike, optional
        a path to save a FLORIS turbine configuration yaml file, optionally

    Returns
    -------
    dict
        a FLORIS turbine configuration in dictionary form

    Raises
    ------
    TypeError
        if the turbine specification input is not the correct type
    """

    if isinstance(input_turbine_spec, PathLike):
        with open(input_turbine_spec, "r") as file_turbine_spec:
            turbine_spec = load_turbine_spec(file_turbine_spec)
    elif type(input_turbine_spec) == dict:
        turbine_spec = input_turbine_spec
    else:
        raise TypeError(
            "create_FLORIS_yamlfile requires either a dict input or a filename input.\n"
            + f"recieved a {type(input_turbine_spec)}"
        )

    # load speed/power/thrust file
    filename_power_thrust = turbine_spec["performance_data_ccblade"]["power_thrust_csv"]
    pt_raw = np.genfromtxt(filename_power_thrust, delimiter=",").T.tolist()

    # create FLORIS config dict
    turbine_FLORIS = dict()
    turbine_FLORIS["turbine_type"] = turbine_spec["description"]["name"]
    turbine_FLORIS["hub_height"] = turbine_spec["geometry"]["height_hub"]
    turbine_FLORIS["rotor_diameter"] = turbine_spec["geometry"]["diameter_rotor"]
    turbine_FLORIS["TSR"] = turbine_spec["nameplate"]["TSR"]
    # turbine_FLORIS["multi_dimensional_cp_ct"] = True
    # turbine_FLORIS["power_thrust_data_file"] = filename_power_thrust
    turbine_FLORIS["power_thrust_table"] = {
        "cosine_loss_exponent_yaw": turbine_spec["model_specifications"]["FLORIS"][
            "exponent_penalty_yaw"
        ],
        "cosine_loss_exponent_tilt": turbine_spec["model_specifications"]["FLORIS"][
            "exponent_penalty_tilt"
        ],
        "peak_shaving_fraction": turbine_spec["model_specifications"]["FLORIS"][
            "fraction_peak_shaving"
        ],
        "peak_shaving_TI_threshold": 0.0,
        "ref_air_density": turbine_spec["performance_data_ccblade"][
            "density_ref_cp_ct"
        ],
        "ref_tilt": turbine_spec["performance_data_ccblade"]["tilt_ref_cp_ct"],
        "wind_speed": pt_raw[0],
        "power": (
            0.5
            * turbine_spec["performance_data_ccblade"]["density_ref_cp_ct"]
            * (np.pi / 4.0 * turbine_spec["geometry"]["diameter_rotor"] ** 2)
            * np.array(pt_raw[0]) ** 3
            * pt_raw[1]
            / 1e3
        ).tolist(),
        "thrust_coefficient": pt_raw[2],
    }

    # If an export filename is given, write it out
    if filename_turbine_FLORIS is not None:
        with open(filename_turbine_FLORIS, "w") as file_turbine_FLORIS:
            yaml.safe_dump(turbine_FLORIS, file_turbine_FLORIS)

    return copy.deepcopy(turbine_FLORIS)
