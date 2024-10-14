import numpy as np
import openmdao.api as om

import pytest

import windard.layout.gridfarm as gridfarm


class TestGridFarm:

    def setup_method(self):

        self.modeling_options = {
            "farm": {
                "N_turbines": 25,
            },
            "turbine": {
                "geometry": {
                    "diameter_rotor": 130.0,
                }
            },
        }

        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        # # BEGIN DEBUG!!!!!
        # self.model.add_subsystem(
        #   "viz",
        #   viz.OutputLayout(modeling_options=self.modeling_options),
        #   promotes=["*"],
        # )
        # # END DEBUG!!!!!
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling_options has what we need for the layout
        assert "modeling_options" in [k for k, _ in self.gf.options.items()]

        assert "farm" in self.gf.options["modeling_options"].keys()
        assert "N_turbines" in self.gf.options["modeling_options"]["farm"].keys()

        assert "turbine" in self.gf.options["modeling_options"].keys()
        assert "geometry" in self.gf.options["modeling_options"]["turbine"].keys()
        assert (
            "diameter_rotor"
            in self.gf.options["modeling_options"]["turbine"]["geometry"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.gf.list_inputs()]
            for var_to_check in [
                "spacing_primary",
                "spacing_secondary",
                "angle_orientation",
                "angle_skew",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.gf.list_outputs()]
            for var_to_check in [
                "x_turbines",
                "y_turbines",
                "spacing_effective_primary",
                "spacing_effective_secondary",
            ]:
                assert var_to_check in output_list

    def test_compute_squarefarm(self):

        spacing1 = 5.0
        spacing2 = 3.0

        self.prob.set_val("gridfarm.spacing_primary", spacing1)
        self.prob.set_val("gridfarm.spacing_secondary", spacing2)
        self.prob.set_val("gridfarm.angle_orientation", 0.0)
        self.prob.set_val("gridfarm.angle_skew", 0.0)

        self.prob.run_model()

        x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
        y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
        X, Y = np.meshgrid(x_turbines, y_turbines)

        assert np.all(np.isclose(self.prob.get_val("gridfarm.x_turbines"), X.flatten()))
        assert np.all(np.isclose(self.prob.get_val("gridfarm.y_turbines"), Y.flatten()))

    def test_compute_rotatedfarm(self):

        spacing1 = 4.0
        spacing2 = 3.0
        angle_orientation = 7.5
        angle_skew = 0.0

        self.prob.set_val("gridfarm.spacing_primary", spacing1)
        self.prob.set_val("gridfarm.spacing_secondary", spacing2)
        self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
        self.prob.set_val("gridfarm.angle_skew", angle_skew)

        self.prob.run_model()

        x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
        y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
        X, Y = np.meshgrid(x_turbines, y_turbines)
        Xr = (
            np.cos(np.radians(angle_orientation)) * X
            + np.sin(np.radians(angle_orientation)) * Y
        )
        Yr = (
            -np.sin(np.radians(angle_orientation)) * X
            + np.cos(np.radians(angle_orientation)) * Y
        )

        assert np.all(
            np.isclose(self.prob.get_val("gridfarm.x_turbines"), Xr.flatten())
        )
        assert np.all(
            np.isclose(self.prob.get_val("gridfarm.y_turbines"), Yr.flatten())
        )

    def test_compute_skewedfarm(self):

        spacing1 = 3.0
        spacing2 = 5.0
        angle_orientation = 0.0
        angle_skew = 12.5

        self.prob.set_val("gridfarm.spacing_primary", spacing1)
        self.prob.set_val("gridfarm.spacing_secondary", spacing2)
        self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
        self.prob.set_val("gridfarm.angle_skew", angle_skew)

        self.prob.run_model()

        x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
        y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
        X, Y = np.meshgrid(x_turbines, y_turbines)
        Xr, Yr = X, Y
        Xs = Xr
        Ys = Yr - Xr * np.tan(np.radians(angle_skew))

        assert np.all(
            np.isclose(self.prob.get_val("gridfarm.x_turbines"), Xs.flatten())
        )
        assert np.all(
            np.isclose(self.prob.get_val("gridfarm.y_turbines"), Ys.flatten())
        )

    def test_compute_rotatedskewedfarm(self):

        spacing1 = 3.0
        spacing2 = 4.0
        angle_orientation = 15.0
        angle_skew = 10.0

        self.prob.set_val("gridfarm.spacing_primary", spacing1)
        self.prob.set_val("gridfarm.spacing_secondary", spacing2)
        self.prob.set_val("gridfarm.angle_orientation", angle_orientation)
        self.prob.set_val("gridfarm.angle_skew", angle_skew)

        self.prob.run_model()

        x_turbines = spacing2 * 130.0 * np.arange(-2, 2 + 1, 1)
        y_turbines = spacing1 * 130.0 * np.arange(-2, 2 + 1, 1)
        X, Y = np.meshgrid(x_turbines, y_turbines)
        Xs = X
        Ys = -np.tan(np.radians(angle_skew)) * X + Y
        Xr = (
            np.cos(np.radians(angle_orientation)) * Xs
            + np.sin(np.radians(angle_orientation)) * Ys
        )
        Yr = (
            -np.sin(np.radians(angle_orientation)) * Xs
            + np.cos(np.radians(angle_orientation)) * Ys
        )

        assert np.all(
            np.isclose(self.prob.get_val("gridfarm.x_turbines"), Xr.flatten())
        )
        assert np.all(
            np.isclose(self.prob.get_val("gridfarm.y_turbines"), Yr.flatten())
        )


class TestGridFarmLanduse:

    def setup_method(self):

        self.N_turbines = 25
        self.D_rotor = 130.0
        self.modeling_options = {
            "farm": {
                "N_turbines": self.N_turbines,
            },
            "turbine": {
                "geometry": {
                    "diameter_rotor": self.D_rotor,
                }
            },
        }

        self.model = om.Group()
        self.gf = self.model.add_subsystem(
            "gridfarm",
            gridfarm.GridFarmLayout(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        self.lu = self.model.add_subsystem(
            "gflu",
            gridfarm.GridFarmLanduse(modeling_options=self.modeling_options),
            promotes=["*"],
        )
        # # BEGIN DEBUG!!!!!
        # self.model.add_subsystem(
        #   "viz",
        #   viz.OutputLayout(modeling_options=self.modeling_options),
        #   promotes=["*"],
        # )
        # # END DEBUG!!!!!
        self.prob = om.Problem(self.model)
        self.prob.setup()

    def test_setup(self):
        # make sure the modeling_options has what we need for the layout
        assert "modeling_options" in [k for k, _ in self.lu.options.items()]

        assert "farm" in self.lu.options["modeling_options"].keys()
        assert "N_turbines" in self.lu.options["modeling_options"]["farm"].keys()

        assert "turbine" in self.lu.options["modeling_options"].keys()
        assert "geometry" in self.lu.options["modeling_options"]["turbine"].keys()
        assert (
            "diameter_rotor"
            in self.lu.options["modeling_options"]["turbine"]["geometry"].keys()
        )

        # context manager to spike the warning since we aren't running the model yet
        with pytest.warns(Warning) as warning:
            # make sure that the outputs in the component match what we planned
            input_list = [k for k, v in self.lu.list_inputs()]
            for var_to_check in [
                "spacing_primary",
                "spacing_secondary",
                "angle_orientation",
                "angle_skew",
                "distance_layback_diameters",
            ]:
                assert var_to_check in input_list

            # make sure that the outputs in the component match what we planned
            output_list = [k for k, v in self.lu.list_outputs()]
            for var_to_check in [
                "area_tight",
                "area_aligned_parcel",
                "area_compass_parcel",
            ]:
                assert var_to_check in output_list

    def test_compute_squarefarm(self):

        spacing1 = 5.0
        spacing2 = 3.0

        self.prob.set_val("gridfarm.spacing_primary", spacing1)
        self.prob.set_val("gridfarm.spacing_secondary", spacing2)
        self.prob.set_val("gridfarm.angle_orientation", 0.0)
        self.prob.set_val("gridfarm.angle_skew", 0.0)

        self.prob.run_model()

        A_ref = (
            (np.sqrt(self.N_turbines) - 1) ** 2
            * spacing1
            * spacing2
            * self.D_rotor**2
            / (1000.0) ** 2
        )

        print(self.prob.get_val("gflu.area_tight"), A_ref)
        assert np.isclose(self.prob.get_val("gflu.area_tight"), A_ref)
