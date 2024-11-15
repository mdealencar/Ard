import openmdao.api as om

import windard.layout.gridfarm as gridfarm
import windard.farm_aero.floris as farmaero_floris
import windard.cost.wisdem_wrap as cost_wisdem


def create_setup_OM_problem(
    modeling_options,
    wind_rose = None,
    aero_backend = "FLORIS",
    layout_type = "gridfarm",
):
    if layout_type != "gridfarm":
        raise NotImplementedError(f"layout type {layout_type} is not implemented yet. -cfrontin")
    if aero_backend != "FLORIS":
        raise NotImplementedError(f"aerodynamic backend {aero_backend} is not implemented yet. -cfrontin")
    if wind_rose is None:
        raise NotImplementedError("this wind rose configuration is not implemented. -cfrontin")

    # create the OpenMDAO model
    model = om.Group()
    group_layout2aep = om.Group()
    group_layout2aep.add_subsystem(  # layout component
        "layout",
        gridfarm.GridFarmLayout(modeling_options=modeling_options),
        promotes=["*"],
    )
    group_layout2aep.add_subsystem(  # landuse component
        "landuse",
        gridfarm.GridFarmLanduse(modeling_options=modeling_options),
        promotes_inputs=["*"],
    )
    group_layout2aep.add_subsystem(  # FLORIS AEP component
        "aepFLORIS",
        farmaero_floris.FLORISAEP(
            modeling_options=modeling_options,
            wind_rose=wind_rose,
            case_title="letsgo",
        ),
        # promotes=["AEP_farm"],
        promotes=["x_turbines", "y_turbines", "AEP_farm"],
    )
    group_layout2aep.approx_totals(method="fd", step=1e-3, form="central", step_calc="rel_avg")
    model.add_subsystem(
        "layout2aep",
        group_layout2aep,
        promotes=[
            "angle_orientation",
            "angle_skew",
            "spacing_primary",
            "spacing_secondary",
            "spacing_effective_primary",
            "spacing_effective_secondary",
            "AEP_farm",
        ],
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
    model.connect("AEP_farm", "financese.plant_aep_in")
    model.connect("landbosse.bos_capex_kW", "financese.bos_per_kW")

    # build out the problem based on this model
    prob = om.Problem(model)
    prob.setup()

    # return the problem
    return prob

