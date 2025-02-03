import copy
from os import PathLike
from pathlib import Path
import yaml

import numpy as np

from wisdem.inputs.validation import load_yaml

def smoothmin(target, alpha_smoothmin=10.0):
    if False:
        kernel_smoothmin = alpha_smoothmin * np.array(target)  # basic
    elif False:
        kernel_smoothmin = np.maximum(
            alpha_smoothmin * np.array(target), -500.0
        )  # target buffer to prevent overflow
    else:
        kernel_overflow_max = -500.0
        target = alpha_smoothmin * np.array(target)
        eps_smu = 1e-6
        kernel_smoothmin = 0.5 * (
            kernel_overflow_max
            + np.array(target)
            + np.sqrt((kernel_overflow_max - target) ** 2 + eps_smu)
        )
    return 1 / alpha_smoothmin * np.log(np.mean(np.exp(kernel_smoothmin)))

def smooth_max(x:np.ndarray, s:float=10.0) -> float:
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
        s (float, optional): alpha for smooth max function. Defaults to 10.0. 
            Larger values of `s` lead to more accurate results, but reduce the smoothness 
            of the output values.

    Returns:
        float: the smooth max of the provided `x` list
    """

    # get the maximum value and the index of maximum value
    max_ind = np.argmax(x)
    max_val = x[max_ind]
    
    # get the indices of x
    indices = np.arange(0, len(x), dtype=int)

    # remove the index of the maximum value
    # indices = np.delete(indices, max_ind)

    # LogSumExp with smoothing factor s
    # import pdb; pdb.set_trace()
    exponential = np.exp(s*(x[indices != max_ind] - max_val))
    r = (np.log(1.0 + np.sum([exponential])) + s*max_val)/s

    return r

def smooth_min(x:np.ndarray, s:float=10.0) -> float:
    """ Finds smooth min using the `smooth_max` function

    Args:
        x (list): list of values to be compared
        s (float, optional): alpha for smooth min function. Defaults to 10.0. 
            Larger values of `s` lead to more accurate results, but reduce the smoothness 
            of the output values.

    Returns:
        float: the smooth min of the provided `x` list
    """

    return -smooth_max(x=-x, s=s)

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
