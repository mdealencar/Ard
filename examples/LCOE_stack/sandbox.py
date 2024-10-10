import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

from wisdem.inputs.validation import load_yaml

import windard.utils
import windard.wind_query as wq
import windard.layout.gridfarm as gridfarm
import windard.farm_aero.floris as farmaero_floris
import windard.cost.wisdem_wrap as cost_wisdem

# create the wind query
wind_rose_wrg = floris.wind_data.WindRoseWRG(
    os.path.join(
        os.path.split(floris.__file__)[0],
        "..",
        "examples",
        "examples_wind_resource_grid",
        "wrg_example.wrg",
    )
)
wind_rose_wrg.set_wd_step(1.0)
wind_rose_wrg.set_wind_speeds(np.arange(0, 30, 0.5)[1:])
wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
wind_query = wq.WindQuery.from_FLORIS_WindData(wind_rose)

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
        "FLORIS_turbine_library",
        "IEA-3p4-130-RWT.yaml",
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
        "N_turbines": 25,
    },
    "turbine": data_turbine,
    "FLORIS": {
        "filename_tool_config": filename_floris_config,
    },
}

# create the OpenMDAO model
model = om.Group()
model.add_subsystem(
    "layout",
    gridfarm.GridFarmLayout(modeling_options=modeling_options),
    promotes=["*"],
)
model.add_subsystem(
    "aepFLORIS",
    farmaero_floris.FLORISAEP(
        modeling_options=modeling_options,
        wind_rose=wind_rose,
        case_title="letsgo",
    ),
    promotes=["x_turbines", "y_turbines"],
)
model.add_subsystem(
    "tcc",
    cost_wisdem.TurbineCapitalCosts(),
    promotes_inputs=["turbine_number", "machine_rating", "tcc_per_kW", "offset_tcc_per_kW",],
)
model.add_subsystem(
    "landbosse",
    cost_wisdem.LandBOSSE(),
)
model.add_subsystem(
    "opex",
    cost_wisdem.OperatingExpenses(),
    promotes_inputs=["turbine_number", "machine_rating", "opex_per_kW",],
)
model.connect("spacing_effective_primary", "landbosse.turbine_spacing_rotor_diameters")
model.connect("spacing_effective_secondary", "landbosse.row_spacing_rotor_diameters")

model.add_subsystem(
    "financese",
    cost_wisdem.PlantFinance(),
    promotes_inputs=["turbine_number", "machine_rating", "tcc_per_kW", "offset_tcc_per_kW", "opex_per_kW",]
)
model.connect("aepFLORIS.AEP_farm", "financese.plant_aep_in")
model.connect("landbosse.bos_capex_kW", "financese.bos_per_kW")

prob = om.Problem(model)
prob.setup()

# set the latent variables for using the cost models
prob.set_val("landbosse.num_turbines", modeling_options["farm"]["N_turbines"])
prob.set_val("turbine_number", int(modeling_options["farm"]["N_turbines"]))
prob.set_val(
    "landbosse.turbine_rating_MW",
    modeling_options["turbine"]["nameplate"]["power_rated"] * 1.0e3,
)
prob.set_val(
    "machine_rating",
    modeling_options["turbine"]["nameplate"]["power_rated"] * 1.0e3,
)

# inputs to LandBOSSE
prob["landbosse.hub_height_meters"] = modeling_options["turbine"]["geometry"][
    "height_hub"
]
prob["landbosse.wind_shear_exponent"] = modeling_options["turbine"]["costs"][
    "wind_shear_exponent"
]
prob["landbosse.rotor_diameter_m"] = modeling_options["turbine"]["geometry"][
    "diameter_rotor"
]
prob["landbosse.number_of_blades"] = modeling_options["turbine"]["geometry"][
    "num_blades"
]
prob["landbosse.rated_thrust_N"] = modeling_options["turbine"]["costs"][
    "rated_thrust_N"
]
prob["landbosse.gust_velocity_m_per_s"] = modeling_options["turbine"]["costs"][
    "gust_velocity_m_per_s"
]
prob["landbosse.blade_surface_area"] = modeling_options["turbine"]["costs"][
    "blade_surface_area"
]
prob["landbosse.tower_mass"] = modeling_options["turbine"]["costs"]["tower_mass"]
prob["landbosse.nacelle_mass"] = modeling_options["turbine"]["costs"]["nacelle_mass"]
prob["landbosse.hub_mass"] = modeling_options["turbine"]["costs"]["hub_mass"]
prob["landbosse.blade_mass"] = modeling_options["turbine"]["costs"]["blade_mass"]
prob["landbosse.foundation_height"] = modeling_options["turbine"]["costs"][
    "foundation_height"
]
prob["landbosse.commissioning_pct"] = modeling_options["turbine"]["costs"][
    "commissioning_pct"
]
prob["landbosse.decommissioning_pct"] = modeling_options["turbine"]["costs"][
    "decommissioning_pct"
]
prob["landbosse.trench_len_to_substation_km"] = modeling_options["turbine"]["costs"][
    "trench_len_to_substation_km"
]
prob["landbosse.distance_to_interconnect_mi"] = modeling_options["turbine"]["costs"][
    "distance_to_interconnect_mi"
]
prob["landbosse.interconnect_voltage_kV"] = modeling_options["turbine"]["costs"][
    "interconnect_voltage_kV"
]

# inputs to PlantFinanceSE
prob["tcc_per_kW"] = modeling_options["turbine"]["costs"]["tcc_per_kW"]
prob["opex_per_kW"] = modeling_options["turbine"]["costs"]["opex_per_kW"]

# set up the working/design variables
prob.set_val("spacing_primary", 7.0)
prob.set_val("spacing_secondary", 7.0)
prob.set_val("angle_orientation", 0.0)
prob.set_val("angle_skew", 0.0)

# run the model
prob.run_model()

# get and print the AEP
AEP_val = float(prob.get_val("aepFLORIS.AEP_farm", units="GW*h")[0])
CapEx_val = float(prob.get_val("tcc.tcc", units="MUSD")[0])
BOS_val = float(prob.get_val("landbosse.total_capex", units="MUSD")[0])
OpEx_val = float(prob.get_val("opex.opex", units="MUSD/yr")[0])
LCOE_val = float(prob.get_val("financese.lcoe", units="USD/MW/h")[0])
print(f"AEP: {AEP_val:.2f}")
print(f"ICC: M${CapEx_val+BOS_val:.2f}")
print(f"\tCapEx: M${CapEx_val:.2f}")
print(f"\tBOS: M${BOS_val:.2f}")
print(f"OpEx/yr.: M${OpEx_val:.2f}")
print(f"LCOE: ${LCOE_val:.2f}/MWh")
# FIN!
