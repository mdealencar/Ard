
# Component types

The design intention of `windArd` is to offer a principled, modular, extensible wind farm layout optimization tool.

In order to balance focus with the modularity and extensibility intended for the code, we will classify different types of components, and build "base `windArd`" with a set of default components.
Each type of component will be defined below, and each type of component will have a template parent class which can be used to derive custom user-defined components to serve as drop in replacements in `windArd`.

## Layout DV Components (`layout`)

Wind farm layout optimization is a significantly challenging problem for global optimization, due to the existence of many local minima.
One strategy for reducing the dimensionality of the design space is the use of layout models.
`layout` components are for connecting some reduced layout variables to (`x_turbines`, `y_turbines`) variables that explicitly describe the layout of a farm for computing the farm aerodynamics.

**tl;dr:** `layout` components map from a simplified parameter set to Cartesian farm coordinates

## Farm Aero Components (`farm_aero`)

Fundamentally, `farm_aero` components will take in a set of farm layout design variables, in terms of `x_turbines` and `y_turbines` components of turbine locations, and potentially with some control command input, namely `yaw`.

In addition to these design variables, the turbine definitions to be used and some (possibly comprehensive) set of wind conditions to be queried will also be provided to a given `farm_aero` component.

The result of a `farm_aero` component will be a power or energy production quantity of interest.
Typically, these will be a power output estimate for the set of provided conditions or annual energy production estimate for the farm given the wind resource.

**tl;dr:** `farm_aero` components map from a farm layouts and possibly control settings to some measure of farm power production



<!-- FIN! -->
