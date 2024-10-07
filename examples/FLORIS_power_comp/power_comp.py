import os

import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

from wisdem.inputs.validation import load_yaml

import windard.utils
import windard.wind_query as wq
import windard.farm_aero_comp.floris as farmaero_floris

# create the wind query
directions = np.linspace(0.0, 360.0, 101)
speeds = np.linspace(0.0, 30.0, 101)[1:]
WS, WD = np.meshgrid(speeds, directions)
wind_query = wq.WindQuery(WD.flatten(), WS.flatten(), TIs=0.06)

# create the farm layout specification
farm_spec = {}
farm_spec["xD_farm"], farm_spec["yD_farm"] = [
    7 * v.flatten() for v in np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
]
if False:
  plt.scatter(farm_spec["xD_farm"], farm_spec["yD_farm"])
  plt.show()

# specify the configuration/specification files to use
filename_turbine_spec = os.path.abspath(
    os.path.join(
        "data",
        "turbine_spec_IEA-3p4-130-RWT.yaml",
    )
)  # toolset generalized turbine specification
filename_turbine_FLORIS = os.path.abspath(
    os.path.join(
        "data",
        "FLORISturbine_IEA-3p4-130-RWT.yaml",
    )
)  # toolset generalized turbine specification
filename_floris_config = os.path.abspath(
    os.path.join(
        "data",
        "FLORIS.yaml",
    )
)  # default FLORIS config for the project
# create a FLORIS yaml to conform to the config/spec files above
windard.utils.create_FLORIS_yamlfile(filename_turbine_spec, filename_turbine_FLORIS)
# load the turbine specification
data_turbine = load_yaml(filename_turbine_spec)

# set up the modeling options
modeling_options = {
    "farm": {
        "N_turbines": len(farm_spec["xD_farm"]),
    },
    "turbine": data_turbine,
    "FLORIS": {
        "filename_tool_config": filename_floris_config,
    },
}

# create the OpenMDAO model
model = om.Group()
model.add_subsystem(
  "batchFLORIS",
  farmaero_floris.FLORISBatchPower(
    modeling_options=modeling_options,
    wind_query=wind_query,
    case_title="letsgo",
  ),
)

prob = om.Problem(model)
prob.setup()

prob.set_val("batchFLORIS.x", 130.0*farm_spec["xD_farm"])
prob.set_val("batchFLORIS.y", 130.0*farm_spec["yD_farm"])

prob.run_model()

print(prob.get_val("batchFLORIS.power_farm", units="GW"))

plt.contourf(
  WD, WS,
  prob.get_val("batchFLORIS.power_farm", units="GW").reshape(WD.shape),
)
plt.show()

### FIN!
