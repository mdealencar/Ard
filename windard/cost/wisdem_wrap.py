import warnings

import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance as PlantFinance_orig
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE as LandBOSSE_orig
from wisdem.inputs.validation import load_yaml


# this wrapper should sandbag warnings
class LandBOSSE(LandBOSSE_orig):
    def setup(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        with warnings.catch_warnings():
            return super().setup()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        warnings.filterwarnings("ignore", category=FutureWarning)
        with warnings.catch_warnings():
            return super().compute(inputs, outputs, discrete_inputs, discrete_outputs)


# this wrapper should sandbag warnings
class PlantFinance(PlantFinance_orig):
    def setup(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        with warnings.catch_warnings():
            return super().setup()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        warnings.filterwarnings("ignore", category=FutureWarning)
        with warnings.catch_warnings():
            return super().compute(inputs, outputs, discrete_inputs, discrete_outputs)


class TurbineCapitalCosts(om.ExplicitComponent):
    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("tcc_per_kW", 0.0, units="USD/kW")
        self.add_input("offset_tcc_per_kW", 0.0, units="USD/kW")
        self.add_discrete_input("turbine_number", 0)
        self.add_output("tcc", 0.0, units="USD")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        tcc_per_kW = inputs["tcc_per_kW"] + inputs["offset_tcc_per_kW"]
        outputs["tcc"] = n_turbine * tcc_per_kW * t_rating


class OperatingExpenses(om.ExplicitComponent):
    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("opex_per_kW", 0.0, units="USD/kW/yr")
        self.add_discrete_input("turbine_number", 0)
        self.add_output("opex", 0.0, units="USD/yr")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        opex_per_kW = inputs["opex_per_kW"]
        outputs["opex"] = n_turbine * opex_per_kW * t_rating


# class WindPark(om.Group):
#     # Openmdao group to run the cost analysis of a wind park
#     def setup(self):
#
#         costs_ivc = self.add_subsystem("costs", om.IndepVarComp())
#         costs_ivc.add_discrete_output(
#             "turbine_number", val=0, desc="Number of turbines at plant"
#         )
#         costs_ivc.add_output(
#             "turbine_rating", val=0.0, units="W", desc="Rating of the turbine."
#         )
#         costs_ivc.add_output(
#             "plant_aep",
#             val=0.0,
#             units="W*h",
#             desc="Annual energy production of the plant.",
#         )
#
#         # self.add_subsystem("farm_aep", FarmAEP())
#         self.add_subsystem("landbosse", LandBOSSE())
#         self.add_subsystem("financese", PlantFinance())
#
#         self.connect("landbosse.bos_capex_kW", "financese.bos_per_kW")
#         self.connect(
#             "costs.turbine_number",
#             ["landbosse.num_turbines", "financese.turbine_number"],
#         )
#         self.connect(
#             "costs.turbine_rating",
#             ["landbosse.turbine_rating_MW", "financese.machine_rating"],
#         )
#         self.connect("costs.plant_aep", "financese.plant_aep_in")
#         # self.connect("farm_aep.aep", "financese.plant_aep_in")


# def initialize_park_prob(park_opt, path2yaml):
#
#     yaml_inputs = load_yaml(path2yaml)
#
#     # Common inputs
#     park_opt["costs.turbine_number"] = yaml_inputs["turbine_number"]
#     park_opt["costs.turbine_rating"] = yaml_inputs["turbine_rating"]
#     park_opt["costs.plant_aep"] = yaml_inputs["turbine_aep"] * (
#         park_opt["costs.turbine_number"] * 0.85
#     )
#
#     # Inputs to LandBOSSE
#     park_opt["landbosse.hub_height_meters"] = yaml_inputs["hub_height_meters"]
#     park_opt["landbosse.wind_shear_exponent"] = yaml_inputs["wind_shear_exponent"]
#     park_opt["landbosse.rotor_diameter_m"] = yaml_inputs["rotor_diameter_m"]
#     park_opt["landbosse.number_of_blades"] = yaml_inputs["number_of_blades"]
#     park_opt["landbosse.rated_thrust_N"] = yaml_inputs["rated_thrust_N"]
#     park_opt["landbosse.gust_velocity_m_per_s"] = yaml_inputs["gust_velocity_m_per_s"]
#     park_opt["landbosse.blade_surface_area"] = yaml_inputs["blade_surface_area"]
#     park_opt["landbosse.tower_mass"] = yaml_inputs["tower_mass"]
#     park_opt["landbosse.nacelle_mass"] = yaml_inputs["nacelle_mass"]
#     park_opt["landbosse.hub_mass"] = yaml_inputs["hub_mass"]
#     park_opt["landbosse.blade_mass"] = yaml_inputs["blade_mass"]
#     park_opt["landbosse.foundation_height"] = yaml_inputs["foundation_height"]
#     park_opt["landbosse.turbine_spacing_rotor_diameters"] = yaml_inputs[
#         "turbine_spacing_rotor_diameters"
#     ]
#     park_opt["landbosse.row_spacing_rotor_diameters"] = yaml_inputs[
#         "row_spacing_rotor_diameters"
#     ]
#     park_opt["landbosse.commissioning_pct"] = yaml_inputs["commissioning_pct"]
#     park_opt["landbosse.decommissioning_pct"] = yaml_inputs["decommissioning_pct"]
#     park_opt["landbosse.trench_len_to_substation_km"] = yaml_inputs[
#         "trench_len_to_substation_km"
#     ]
#     park_opt["landbosse.distance_to_interconnect_mi"] = yaml_inputs[
#         "distance_to_interconnect_mi"
#     ]
#     park_opt["landbosse.interconnect_voltage_kV"] = yaml_inputs[
#         "interconnect_voltage_kV"
#     ]
#
#     # Inputs to PlantFinanceSE
#     park_opt["financese.tcc_per_kW"] = yaml_inputs["tcc_per_kW"]
#     park_opt["financese.opex_per_kW"] = yaml_inputs["opex_per_kW"]
#
#     return park_opt
