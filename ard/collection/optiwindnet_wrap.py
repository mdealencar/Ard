import numpy as np

from optiwindnet.mesh import make_planar_embedding
from optiwindnet.interarraylib import L_from_site
from optiwindnet.heuristics import EW_presolver
from optiwindnet.MILP import solver_factory, ModelOptions

from . import templates


def optiwindnet_wrapper(
    XY_turbines: np.ndarray,
    XY_substations: np.ndarray,
    XY_boundaries: np.ndarray | None,
    name_case: str,
    max_turbines_per_string: int,
    solver_name: str = "highs",
    time_limit: int = 60,
    mip_gap: float = 0.005,
    solver_options: dict | None = None,
    verbose: bool = False,
):
    """Simple wrapper to run OptiWindNet to get a cable layout

    Args:
        XY_turbines (np.ndarray): x and y positions of turbines (easting and northing)
        XY_substations (np.ndarray): x and y positions of substations (easting and northing)
        XY_boundaries (np.ndarray): x and y locations of boundary nodes (easting and northing)
        name_case (str):  what to name the case
        max_turbines_per_string (int): maximum number of turbines per cable string
        solver_name (str, optional): solver to use. Defaults to "highs".
        time_limit: maximum time (s) to allow the solver to run.
        mip_gap: relative distance to stop the search (from incumbent solution
            to best bound).
        solver_options (dict, optional): solver options. Defaults to None.
        verbose (bool, optional): whether to print information. Defaults to False.

    Returns:
        result: MILP solver result
        S: OptiWindNet solution topology
        G: OptiWindNet route-set for the solution
    """
    pass


class optiwindnetCollection(templates.CollectionTemplate):
    """
    Component class for modeling optiwindnet-optimized energy collection systems.

    A component class to make a heuristic-based optimized energy collection and
    management system using optiwindnet! Inherits the interface from
    `templates.CollectionTemplate`.

    Options
    -------
    modeling_options : dict

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

        self.declare_partials(
            ["total_length_cables"],
            ["x_turbines", "y_turbines", "x_substations", "y_substations"],
            method="exact",
        )

    def compute(
        self,
        inputs,
        outputs,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        Computation for the OptiWindNet collection system design
        """

        max_turbines_per_string = self.modeling_options["collection"][
            "max_turbines_per_string"
        ]
        solver_name = self.modeling_options["collection"]["solver_name"]

        # get a graph representing the updated location
        L = L_from_site(**self.site_from_inputs(inputs))

        # create planar embedding and set of available links
        P, A = make_planar_embedding(L)

        # presolve
        S_warm = EW_presolver(A, capacity=max_turbines_per_string)

        # do the branch-and-bound MILP search
        solver = solver_factory(solver_name)
        solver.set_problem(
            P,
            A,
            max_turbines_per_string,
            ModelOptions(**self.modeling_options["collection"]["model_options"]),
            warmstart=S_warm,
        )
        result = solver.solve(**self.modeling_options["collection"]["solver_options"])
        S, G = solver.get_solution()

        # extract the outputs
        self.graph = G
        # ATTENTION: The number of edges in G may be greater than T, because
        # of contours and detours.
        num_edges = G.number_of_edges()
        lengths = np.empty((num_edges,), dtype=np.float64)
        loads = np.empty((num_edges,), dtype=np.float64)

        for i, (_, _, edge_data) in enumerate(G.edges(data=True)):
            lengths[i] = edge_data["length"]
            loads[i] = edge_data["load"]

        # pack and ship
        discrete_outputs["length_cables"] = lengths
        discrete_outputs["load_cables"] = loads
        discrete_outputs["max_load_cables"] = loads.max().item()
        outputs["total_length_cables"] = lengths.sum().item()

    def compute_partials(self, inputs, J, discrete_inputs=None):

        # re-load the key variables back as locals
        G = self.graph
        T = G.graph["T"]
        R = G.graph["R"]
        VertexC = G.graph["VertexC"]
        gradients = np.zeros_like(VertexC)

        fnT = G.graph.get("fnT")
        if fnT is not None:
            _u, _v = fnT[np.array(G.edges)].T
        else:
            _u, _v = np.array(G.edges).T
        vec = VertexC[_u] - VertexC[_v]
        norm = np.hypot(*vec.T)
        # suppress the contributions of zero-length edges
        norm[np.isclose(norm, 0.0)] = 1.0
        vec /= norm[:, None]

        np.add.at(gradients, _u, vec)
        np.subtract.at(gradients, _v, vec)

        # wind turbines
        J["total_length_cables", "x_turbines"] = gradients[:T, 0]
        J["total_length_cables", "y_turbines"] = gradients[:T, 1]

        # substations
        J["total_length_cables", "x_substations"] = gradients[-R:, 0]
        J["total_length_cables", "y_substations"] = gradients[-R:, 1]

        return J

    @classmethod
    def site_from_inputs(cls, inputs: dict) -> dict:
        T = len(inputs["x_turbines"])
        R = len(inputs["x_substations"])
        name_case = "farm"
        if "x_borders" in inputs:
            B = len(inputs["x_borders"])
        else:
            B = 0
        VertexC = np.empty((R + T + B, 2), dtype=float)
        VertexC[:T, 0] = inputs["x_turbines"]
        VertexC[:T, 1] = inputs["y_turbines"]
        VertexC[-R:, 0] = inputs["x_substations"]
        VertexC[-R:, 1] = inputs["y_substations"]
        site = dict(
            T=T,
            R=R,
            name=name_case,
            handle=name_case,
            VertexC=VertexC,
        )
        if B > 0:
            VertexC[T:-R, 0] = inputs["x_borders"]
            VertexC[T:-R, 1] = inputs["y_borders"]
            site["B"] = B
            site["border"] = np.arange(T, T + B)
        return site
