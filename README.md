# STL_optimization
Comparison of optimization methods for controller synthesis from STL specifications

## Dependencies

- python3
- numpy
- scipy
- matplotlib
- [scikit-optimize](https://scikit-optimize.github.io/)

## Usage

See `reach_avoid_example.py` for a simple example of usage. 

Scenarios (including robot dynamics, STL specification, etc) should be defined 
following the templates in `example_scenarios.py`. The `cost_function` method 
ultimately defines the cost function which we will optimize over. This is a function
from control inputs to the robustness degree: any number of black-box optimization
algorithms can be applied to this function.

