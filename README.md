
# Ard

`Ard` is an optimization suite for multi-disciplinary optimization of wind farms.

## Name

# TODO
`Ard` is not a typo in the middle of "windward".
It is a portmanteau of "wind" and ["ard"](https://en.wikipedia.org/wiki/Ard_\(plough\)), which is a type of simple and lightweight plow used through the single-digit centuries to prepare a farm for planting.

## Goals

The technical goals of `Ard` are to:
1) allow optimization of wind farm layouts for specific wind resource profiles
1) target wholistic and complex system-level optimization objectives like LCOE and beyond-LCOE metrics
1) naturally incorporate multi-fidelity analyses to efficiently integrate physics-resolving simulation

As a piece of software, the goal of `Ard` is to be:
1) principled: fully documented, and adhering to best-practices for code development
1) modular and extensible: choose the parts you want, skip the ones you don't, build yourself the ones we don't have
1) effective: fully tested and testable at the unit and system level, and built with a derivative-forward approach

These will not always be possible, but they are the goal.

## Prototype design problem

In the following figure, we have a prototype case for Ard:

![`Ard` demonstration image](/assets/ard_xdsm/ard_xdsm.png)

This example shows the variable interactions for an LCOE minimization problem for a farm with a structured layout that can be parametrized by two spacing lengths and two angles.
The problem is also set up to be constrained by the amount of area that is occupied by a wind farm, which can be calculated using the layout parameters.

The layout parameterization is controlled by an optimization loop which controls the layout design variables to minimize LCOE while satisfying a landuse constraint.
These layout design variables are passed to a unit that determines the $(x,y)$ locations of the turbines for the parametrized layout, as well as to an area computation for the land use.
The wind resource is passed in as a set of datapoints to a "de-multiplexer" ("demux") component which takes the single wind resource description and outputs the set of specific wind conditions that should be queried by aerodynamic solvers.
The aerodynamic solvers result in an evaluation of the farm power for a given layout at the set of wind conditions to be queried.
These powers, with the "weights" provided by the demux component allow the integration of AEP.
Finally, the BOS costs can be calculated based on the layout variables, and along with input CapEx and OpEx costs, LCOE can be computed.

## Installation

`Ard` can be installed locally from the source code with `pip` or through a package manager
from PyPI with `pip` or conda-forge with `conda`.
For Windows systems, `conda` is required due to constraints in the WISDEM installation system.
For macOS and Linux, any option is available.

### Local installation from source
To install locally, first ensure that the source code repository is downloaded and configured
to the correct version (or branch or commit).
Then, navigate to the root directory of the source code and run one of the following commands:

```bash
    pip install .           # Install normally
    pip install -e .        # Editable install; changes to the source code affect the installation
    pip install .[dev]      # Developer install; includes dependencies for development
```

<!-- FIN -->
