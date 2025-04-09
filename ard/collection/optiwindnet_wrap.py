import numpy as np

from optiwindnet.mesh import make_planar_embedding as own_make_planar_embedding
from optiwindnet.interarraylib import G_from_S as own_G_from_S
from optiwindnet.interarraylib import L_from_site as own_L_from_site
from optiwindnet.heuristics import EW_presolver as own_EW_presolver
from optiwindnet.pathfinding import PathFinder as OWNPathFinder
from optiwindnet.MILP import pyomo as own_pyomo

from pyomo import environ as pyo
from pyomo.contrib.appsi.solvers import Highs

import ard.collection.templates as templates

import logging

logging.getLogger("optiwindnet").setLevel(logging.INFO)


# custom length calculation
def distance_function(x0, y0, x1, y1):
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def distance_function_deriv(x0, y0, x1, y1):
    return np.array(
        [
            ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5) * (x1 - x0),
            ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5) * (y1 - y0),
            -(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5)) * (x1 - x0),
            -(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-0.5)) * (y1 - y0),
        ]
    )


def optiwindnet_wrapper(
    XY_turbines, XY_substations, XY_boundaries, name_case, capacity
):
    # HIGHS solver
    highs_solver = pyo.SolverFactory("appsi_highs")
    highs_solver.available(), type(highs_solver)

    # start the network definition
    L = own_L_from_site(
        T=len(XY_turbines),
        B=len(XY_boundaries),
        R=len(XY_substations),
        VertexC=np.vstack([XY_turbines, XY_boundaries, XY_substations]),
        border=np.arange(len(XY_turbines), len(XY_turbines) + len(XY_boundaries)),
        name=name_case,
        handle=name_case,
    )

    # create a planar embedding for presolve
    P, A = own_make_planar_embedding(L)

    # presolve
    S = own_EW_presolver(A, capacity=capacity)
    G = own_G_from_S(S, A)

    # create minimum length model
    model = own_pyomo.make_min_length_model(
        A,
        capacity,
        gateXings_constraint=False,
        branching=True,
        gates_limit=False,
    )
    own_pyomo.warmup_model(model, S)

    # create the solver and solve
    time_lim_val = 60  # TODO move to be an option probably
    highs_solver.options.update(
        dict(
            time_limit=time_lim_val,
            mip_rel_gap=0.005,  # TODO ???
        )
    )
    result = highs_solver.solve(model, tee=True)

    # do some postprocessing
    S = own_pyomo.S_from_solution(model, highs_solver, result)
    G = own_G_from_S(S, A)
    H = OWNPathFinder(G, planar=P, A=A).create_detours()

    return result, S, G, H


class optiwindnetCollection(templates.CollectionTemplate):
    """
    Component class for modeling optiwindnet-optimized energy collection systems.

    A component class to make a heuristic-based optimized energy collection and
    management system using optiwindnet! Inherits the interface from
    `templates.CollectionTemplate`.

    Options
    -------
    modeling_options : dict
        a modeling optinos dictionary

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines`
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines`
    x_substations : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the substations,
        with length `N_substations`
    y_substations : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the substations,
        with length `N_substations`

    Outputs
    -------
    length_cables : np.ndarray
        a 1D numpy array that holds the lengths of all of the cables necessary to
        collect energy generated
    load_cables : np.ndarray
        a 1D numpy array that holds the load integer (i.e. total number of
        turbines) collected up to this point of the cable
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

    def setup(self):
        """Setup of OM component."""
        super().setup()

    def setup_partials(self):
        """Setup of OM component gradients."""

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        name_case = "farm"
        capacity = 8  # maximum load on a chain #TODO make the capacity a user input

        # roll up the coordinates into a form that optiwindnet #TODO consider adjusting the buffer (0.25)
        XY_turbines = np.vstack([inputs["x_turbines"], inputs["y_turbines"]]).T
        x_min = np.min(XY_turbines[:, 0]) - 0.25 * np.ptp(XY_turbines[:, 0])
        x_max = np.max(XY_turbines[:, 0]) + 0.25 * np.ptp(XY_turbines[:, 0])
        y_min = np.min(XY_turbines[:, 1]) - 0.25 * np.ptp(XY_turbines[:, 1])
        y_max = np.max(XY_turbines[:, 1]) + 0.25 * np.ptp(XY_turbines[:, 1])
        XY_boundaries = np.array(
            [
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
                [x_max, y_min],
            ]
        )
        XY_substations = np.vstack([inputs["x_substations"], inputs["y_substations"]]).T

        result, S, G, H = optiwindnet_wrapper(
            XY_turbines, XY_substations, XY_boundaries, name_case, capacity
        )

        # extract the outputs
        lengths = []
        loads = []
        edges = H.edges()
        self.graph = H

        for edge in edges:
            lengths.append(edges[edge]["length"])
            loads.append(edges[edge]["load"])

        # pack and ship
        outputs["length_cables"] = np.array(lengths)
        outputs["load_cables"] = np.array(loads)
        outputs["total_length_cables"] = np.sum(outputs["length_cables"])
        outputs["max_load_cables"] = np.max(outputs["load_cables"])

    def compute_partials(self, inputs, J):

        # re-load the key variables back as locals
        XY_turbines = np.vstack([inputs["x_turbines"], inputs["y_turbines"]]).T
        XY_substations = np.vstack([inputs["x_substations"], inputs["y_substations"]]).T
        # print(self.graph)
        H = self.graph
        edges = H.edges()

        J["length_cables", "x_turbines"] = 0.0
        J["length_cables", "y_turbines"] = 0.0
        J["length_cables", "x_substations"] = 0.0
        J["length_cables", "y_substations"] = 0.0
        J["load_cables", "x_turbines"] = 0.0
        J["load_cables", "y_turbines"] = 0.0
        J["load_cables", "x_substations"] = 0.0
        J["load_cables", "y_substations"] = 0.0
        J["total_length_cables", "x_turbines"] = 0.0
        J["total_length_cables", "y_turbines"] = 0.0
        J["total_length_cables", "x_substations"] = 0.0
        J["total_length_cables", "y_substations"] = 0.0
        J["max_load_cables", "x_turbines"] = 0.0
        J["max_load_cables", "y_turbines"] = 0.0
        J["max_load_cables", "x_substations"] = 0.0
        J["max_load_cables", "y_substations"] = 0.0

        for idx_edge, edge in enumerate(edges):
            e0, e1 = edge
            x0, y0 = (
                XY_substations[self.N_substations + e0, :]
                if e0 < 0
                else XY_turbines[e0, :]
            )
            x1, y1 = (
                XY_substations[self.N_substations + e1, :]
                if e1 < 0
                else XY_turbines[e1, :]
            )

            # get the derivative function
            dLdx0, dLdy0, dLdx1, dLdy1 = distance_function_deriv(x0, y0, x1, y1)

            if e0 >= 0:
                J["length_cables", "x_turbines"][idx_edge, e0] -= dLdx0
                J["length_cables", "y_turbines"][idx_edge, e0] -= dLdy0
                J["total_length_cables", "x_turbines"][0, e0] -= dLdx0
                J["total_length_cables", "y_turbines"][0, e0] -= dLdy0
            else:
                J["length_cables", "x_substations"][
                    idx_edge, self.N_substations + e0
                ] -= dLdx0
                J["length_cables", "y_substations"][
                    idx_edge, self.N_substations + e0
                ] -= dLdy0
                J["total_length_cables", "x_substations"][
                    0, self.N_substations + e0
                ] -= dLdx0
                J["total_length_cables", "y_substations"][
                    0, self.N_substations + e0
                ] -= dLdy0
            if e1 >= 0:
                J["length_cables", "x_turbines"][idx_edge, e1] -= dLdx1
                J["length_cables", "y_turbines"][idx_edge, e1] -= dLdy1
                J["total_length_cables", "x_turbines"][0, e1] -= dLdx1
                J["total_length_cables", "y_turbines"][0, e1] -= dLdy1
            else:
                J["length_cables", "x_substations"][
                    idx_edge, self.N_substations + e1
                ] -= dLdx1
                J["length_cables", "y_substations"][
                    idx_edge, self.N_substations + e1
                ] -= dLdy1
                J["total_length_cables", "x_substations"][
                    0, self.N_substations + e1
                ] -= dLdx1
                J["total_length_cables", "y_substations"][
                    0, self.N_substations + e1
                ] -= dLdy1
