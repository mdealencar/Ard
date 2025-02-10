import numpy as np
import matplotlib.pyplot as plt  # DEBUG!!!!!

# import openmdao.api as om

from interarray.importer import load_repository
from interarray.plotting import gplot
from interarray.mesh import make_planar_embedding
from interarray.interarraylib import G_from_S
from interarray.interarraylib import L_from_site
from interarray.heuristics import EW_presolver
from interarray.pathfinding import PathFinder
from interarray.MILP import pyomo as omo

from pyomo import environ as pyo
from pyomo.contrib.appsi.solvers import Highs

import ard.collection.templates as templates

import logging

logging.getLogger("interarray").setLevel(logging.INFO)


class InterarrayCollection(templates.CollectionTemplate):
    """
    Component class for modeling interarray-optimized energy collection systems.

    A component class to make a heuristic-based optimized energy collection and
    management system using interarray! Inherits the interface from
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

        # raise NotImplementedError("IMPLEMENT ME!!!!! -cfrontin")
        self.declare_partials("*", "*", method="fd")  # DEBUG!!!!!

    def compute(self, inputs, outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error!
        """

        name_case = "farm"
        capacity = 8  # maximum load on a chain

        # roll up the coordinates into a form that interarray
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
        print(f"DEBUG!!!!! XY_turbines\n{XY_turbines}")
        print(f"DEBUG!!!!! XY_boundaries\n{XY_boundaries}")
        print(f"DEBUG!!!!! XY_substations\n{XY_substations}")

        # HIGHS solver
        highs_solver = pyo.SolverFactory("appsi_highs")
        highs_solver.available(), type(highs_solver)

        # start the network definition
        print(f"XY_turbines.shape: {XY_turbines.shape}")
        print(f"XY_boundaries.shape: {XY_boundaries.shape}")
        print(f"XY_substations.shape: {XY_substations.shape}")
        L = L_from_site(
            T=len(XY_turbines),
            B=len(XY_boundaries),
            R=len(XY_substations),
            VertexC=np.vstack([XY_turbines, XY_boundaries, XY_substations]),
            border=np.arange(len(XY_turbines), len(XY_turbines) + len(XY_boundaries)),
            name=name_case,
            handle=name_case,
        )

        # create a planar embedding for presolve
        P, A = make_planar_embedding(L)

        # presolve
        S = EW_presolver(A, capacity=capacity)
        G = G_from_S(S, A)

        # create minimum length model
        model = omo.make_min_length_model(
            A,
            capacity,
            gateXings_constraint=False,
            branching=True,
            gates_limit=False,
        )
        omo.warmup_model(model, S)

        # create the solver and solve
        time_lim_val = 60  # move to be an option probably
        highs_solver.options.update(
            dict(
                time_limit=time_lim_val,
                mip_rel_gap=0.005,  # ???
            )
        )
        result = highs_solver.solve(model, tee=True)

        # do some postprocessing
        S = omo.S_from_solution(model, highs_solver, result)
        G = G_from_S(S, A)
        H = PathFinder(G, planar=P, A=A).create_detours()

        # extract the outputs
        lengths = []
        loads = []
        edges = H.edges()
        self.graph = H

        for edge in edges:
            lengths.append(edges[edge]["length"])
            loads.append(edges[edge]["load"])
        outputs["length_cables"] = np.array(lengths)
        outputs["load_cables"] = np.array(loads)
