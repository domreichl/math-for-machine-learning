'''
math_for_machine_learning.py by Dominic Reichl, @domreichl
collects mathematical concepts that are essential for machine learning
'''

import elementary_algebra as ea
import linear_algebra as la
import differential_calculus as dc
import integral_calculus as ic

ea.linear_function(3, 9, (-5, 6), 1, annoInter=False) # degree 1 polynomial
''' plot y = 3x + 9
    data range for x: -5 to 5
    slope length: 1x
    don't plot intercept annotation ''' 
ea.quadratic_function(2, 8, -16, (-20, 21), xInter=True, yInter=False, vertex=True, symLine=False) # degree 2 polynomial
''' plot y = 2x**2 + 8x - 16
    data range for x: -20 to 20
    plot roots (xInter) and vertex, but not y intercept nor line of symmetry '''

print(la.vector_properties(9, 9)) # plot and print magnitude and amplitude of a 2d vector
print(la.vector_addition([9,7], [-1,-2])) # plot and print the sum of two 2d vectors
print(la.vector_multiplication([2,3,1], [1,2,-2])) # multiply two 3d vectors
print(la.matrix_operations([[1,2,3],[4,5,6]], [[6,5,4],[3,2,1]], 'div')) # inversely multiply two 3x3 matrices
la.solve_system_of_equations([[4,2],[5,2]], [[14],[16]]) # 2x+4y=18; 6x+2y=34; find x and y
la.eigenvalues_eigenvectors([[2,0],[0,3]]) # print eigenvalues and eigenvectors of a 2x2 matrix

dc.average_rate_of_change() # introduces the concepts 'secant line' and 'slope'
dc.limits() # introduces the concept of limits
dc.discontinuity() # introduces the concept of a function being non-continuous at a given point
dc.differentiation() # introduces the concept of derivatives
dc.critical_points() # introduces the concept of critical points
dc.partial_derivatives() # introduces the concept of partial derivatives for multivariate functions

ic.integral() # no arguments because f(x) = 3x²+2x+1 and integral limits (0,3) are hardcoded in module
