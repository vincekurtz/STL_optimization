# STL_optimization
Comparison of optimization methods for controller synthesis from STL specifications

## Dependencies

- python3
- numpy
- [scikit-optimize](https://scikit-optimize.github.io/)
- [pySTL](https://github.com/vincekurtz/pySTL)

## Usage

See `reach_avoid_example.py` for a simple example of usage. 

Scenarios (including robot dynamics, STL specification, etc) should be defined 
following the templates in `example_scenarios.py`. The `cost_function` method 
ultimately defines the cost function which we will optimize over. This is a function
from control inputs to the robustness degree: any number of black-box optimization
algorithms can be applied to this function.

`optimization_comparison.py` compares several optimization approaches on one of
the examples in `example_scenarios.py`. It can be used to generate plots like the one shown below:

![comparison figure](comparison.jpg)
