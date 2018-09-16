'''
math_for_machine_learning.py
    by Dominic Reichl, @domreichl
collects mathematical concepts that are essential for machine learning
'''

from elementary_algebra import linear_function # to plot degree 1 polynomial
from elementary_algebra import quadratic_function # to plot degree 2 polynomial

linear_function(3, 9, (-5, 6), 1, annoInter=False)
''' plot y = 3x + 9
    data range for x: -5 to 5
    slope length: 1x
    don't plot intercept annotation '''

    
quadratic_function(2, 8, -16, (-20, 21), xInter=True, yInter=False, vertex=True, symLine=False)
''' plot y = 2x**2 + 8x - 16
    data range for x: -20 to 20
    plot x intercepts and vertex, but not y intercept nor line of symmetry '''
