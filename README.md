# math-for-machine-learning

This Python repository collects mathematical concepts that are essential for machine learning:
- The main file **math_for_machine_learning.py** imports all other files as modules and runs exemplaric function calls.
- The module **elementary_algebra** provides two functions for plotting degree 1 and 2 polynomials: linear with slope and intercepts; quadratic with intercepts, vertex, and line of symmetry.
- The module **linear_algebra** provides
  - three functions for plotting vectors, calculating their magnitude and direction, vector addition, and vector multiplication,
  - a function for processing matrices with six operations (addition, subtraction, negation, transposition, multiplication, and division),
  - a function for solving systems of equations with matrices, and
  - a function for calculating eigenvector-eigenvalue pairs.
- The module **differential_calculus** relies heavily on print outputs and introduces key concepts of calculus including
  - average rate of change, secant line, and slope,
  - limits and discontinuity,
  - derivatives (first and second order), critical points, and partial derivatives.
- The module **integral_calculus** calculates an integral and plots its area under a function.

Required third-party modules are *pandas* for data frames, *matplotlib* for graphs, *numpy* for vectors and matrices, and *scipy* for integrals.

If the scripts in this repository appear messy to you, please consider that I have written them solely for myself to refresh and refine my understanding of these concepts. For better structure and usability, I probably should have written jupyter scripts.
