import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

from wisdem.inputs.validation import load_yaml
from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import windard.utils
import windard.wind_query as wq
import windard.layout.gridfarm as gridfarm
import windard.farm_aero.floris as farmaero_floris
import windard.cost.wisdem_wrap as cost_wisdem

### BEGIN: THINGS TO EVENTUALLY OUTSOURCE TO SHARED FUNCTIONS


def LandBOSSE_setup_latents(prob, modeling_options):

    # get a map from the component variables to the promotion variables
    comp2promotion_map = {
        v[0]: v[-1]["prom_name"]
        for v in prob.model.list_vars(val=False, out_stream=None)
    }

    # set latent/non-design inputs to LandBOSSE using values in modeling_options
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.num_turbines"],
        modeling_options["farm"]["N_turbines"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.turbine_rating_MW"],
        modeling_options["turbine"]["nameplate"]["power_rated"] * 1.0e3,
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.hub_height_meters"],
        modeling_options["turbine"]["geometry"]["height_hub"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.wind_shear_exponent"],
        modeling_options["turbine"]["costs"]["wind_shear_exponent"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.rotor_diameter_m"],
        modeling_options["turbine"]["geometry"]["diameter_rotor"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.number_of_blades"],
        modeling_options["turbine"]["geometry"]["num_blades"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.rated_thrust_N"],
        modeling_options["turbine"]["costs"]["rated_thrust_N"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.gust_velocity_m_per_s"],
        modeling_options["turbine"]["costs"]["gust_velocity_m_per_s"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.blade_surface_area"],
        modeling_options["turbine"]["costs"]["blade_surface_area"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.tower_mass"],
        modeling_options["turbine"]["costs"]["tower_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.nacelle_mass"],
        modeling_options["turbine"]["costs"]["nacelle_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.hub_mass"],
        modeling_options["turbine"]["costs"]["hub_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.blade_mass"],
        modeling_options["turbine"]["costs"]["blade_mass"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.foundation_height"],
        modeling_options["turbine"]["costs"]["foundation_height"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.commissioning_pct"],
        modeling_options["turbine"]["costs"]["commissioning_pct"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.decommissioning_pct"],
        modeling_options["turbine"]["costs"]["decommissioning_pct"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.trench_len_to_substation_km"],
        modeling_options["turbine"]["costs"]["trench_len_to_substation_km"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.distance_to_interconnect_mi"],
        modeling_options["turbine"]["costs"]["distance_to_interconnect_mi"],
    )
    prob.set_val(
        comp2promotion_map["landbosse.landbosse.interconnect_voltage_kV"],
        modeling_options["turbine"]["costs"]["interconnect_voltage_kV"],
    )


def FinanceSE_setup_latents(prob, modeling_options):

    # get a map from the component variables to the promotion variables
    comp2promotion_map = {
        v[0]: v[-1]["prom_name"]
        for v in prob.model.list_vars(val=False, out_stream=None)
    }

    # inputs to PlantFinanceSE
    prob.set_val(
        comp2promotion_map["financese.turbine_number"],
        int(modeling_options["farm"]["N_turbines"]),
    )
    prob.set_val(
        comp2promotion_map["financese.machine_rating"],
        modeling_options["turbine"]["nameplate"]["power_rated"] * 1.0e3,
    )
    prob.set_val(
        comp2promotion_map["financese.tcc_per_kW"],
        modeling_options["turbine"]["costs"]["tcc_per_kW"],
    )
    prob.set_val(
        comp2promotion_map["financese.opex_per_kW"],
        modeling_options["turbine"]["costs"]["opex_per_kW"],
    )


def create_setup_OM_problem(modeling_options):
    # create the OpenMDAO model
    model = om.Group()
    model.add_subsystem(  # layout component
        "layout",
        gridfarm.GridFarmLayout(modeling_options=modeling_options),
        promotes=["*"],
    )
    model.add_subsystem(  # landuse component
        "landuse",
        gridfarm.GridFarmLanduse(modeling_options=modeling_options),
        promotes_inputs=["*"],
    )
    model.add_subsystem(  # FLORIS AEP component
        "aepFLORIS",
        farmaero_floris.FLORISAEP(
            modeling_options=modeling_options,
            wind_rose=wind_rose,
            case_title="letsgo",
        ),
        promotes=["x_turbines", "y_turbines"],
    )
    model.add_subsystem(  # turbine capital costs component
        "tcc",
        cost_wisdem.TurbineCapitalCosts(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "tcc_per_kW",
            "offset_tcc_per_kW",
        ],
    )
    model.add_subsystem(  # LandBOSSE component
        "landbosse",
        cost_wisdem.LandBOSSE(),
    )
    model.add_subsystem(  # operational expenditures component
        "opex",
        cost_wisdem.OperatingExpenses(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "opex_per_kW",
        ],
    )
    model.connect(  # effective primary spacing for BOS
        "spacing_effective_primary", "landbosse.turbine_spacing_rotor_diameters"
    )
    model.connect(  # effective secondary spacing for BOS
        "spacing_effective_secondary", "landbosse.row_spacing_rotor_diameters"
    )

    model.add_subsystem(  # cost metrics component
        "financese",
        cost_wisdem.PlantFinance(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "tcc_per_kW",
            "offset_tcc_per_kW",
            "opex_per_kW",
        ],
    )
    model.connect("aepFLORIS.AEP_farm", "financese.plant_aep_in")
    model.connect("landbosse.bos_capex_kW", "financese.bos_per_kW")

    # build out the problem based on this model
    prob = om.Problem(model)
    prob.setup()

    # return the problem
    return prob


### END: THINGS TO EVENTUALLY OUTSOURCE TO SHARED FUNCTIONS

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

# create the OM problem
prob = create_setup_OM_problem(modeling_options=modeling_options)

if True:

    # setup the latent variables for LandBOSSE and FinanceSE
    LandBOSSE_setup_latents(prob, modeling_options)
    FinanceSE_setup_latents(prob, modeling_options)

    # set up the working/design variables
    prob.set_val("spacing_primary", 7.0)
    prob.set_val("spacing_secondary", 7.0)
    prob.set_val("angle_orientation", 0.0)
    prob.set_val("angle_skew", 0.0)

    # run the model
    prob.run_model()

else:

    # set up the working/design variables
    prob.model.add_design_var("spacing_primary", lower=1.0, upper=13.0)
    prob.model.add_design_var("spacing_secondary", lower=1.0, upper=13.0)
    prob.model.add_design_var("angle_orientation", lower=-90.0, upper=90.0)
    prob.model.add_design_var("angle_skew", lower=-90.0, upper=90.0)
    prob.model.add_objective("financese.lcoe")
    # prob.model.add_objective("landuse.area_tight")

    # setup an optimization
    if False:
        prob.driver = om.pyOptSparseDriver(optimizer="SLSQP")
    elif True:
        prob.driver = NLoptDriver(optimizer="LN_COBYLA")
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
    elif True:
        prob.driver = om.ScipyOptimizeDriver(optimizer="COBYLA")
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
    else:
        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options["max_gen"] = 10  # DEBUG!!!!! short
        prob.driver.options["pop_size"] = 5  # DEBUG!!!!! short
        # prob.driver.options["Pc"] = 0.5
        # prob.driver.options["F"] = 0.5
        prob.driver.options["run_parallel"] = True
        prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]
    prob.setup()

    # setup the latent variables for LandBOSSE and FinanceSE
    LandBOSSE_setup_latents(prob, modeling_options)
    FinanceSE_setup_latents(prob, modeling_options)

    # set up the working/design variables
    prob.set_val("spacing_primary", 7.0)
    prob.set_val("spacing_secondary", 7.0)
    prob.set_val("angle_orientation", 0.0)
    prob.set_val("angle_skew", 0.0)

    prob.run_driver()


# get and print the AEP
AEP_val = float(prob.get_val("aepFLORIS.AEP_farm", units="GW*h")[0])
CapEx_val = float(prob.get_val("tcc.tcc", units="MUSD")[0])
BOS_val = float(prob.get_val("landbosse.total_capex", units="MUSD")[0])
OpEx_val = float(prob.get_val("opex.opex", units="MUSD/yr")[0])
LCOE_val = float(prob.get_val("financese.lcoe", units="USD/MW/h")[0])

print(f"AEP: {AEP_val:.2f} GWh")
print(f"ICC: ${CapEx_val+BOS_val:.2f}M")
print(f"\tCapEx: ${CapEx_val:.2f}M")
print(f"\tBOS: ${BOS_val:.2f}M")
print(f"OpEx/yr.: ${OpEx_val:.2f}M")
print(f"LCOE: ${LCOE_val:.2f}/MWh")


# FIN!
