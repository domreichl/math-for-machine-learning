'''
math_for_machine_learning.py by Dominic Reichl, @domreichl
collects mathematical concepts that are essential for machine learning
'''

from elementary_algebra import linear_function # for plotting degree 1 polynomial
from elementary_algebra import quadratic_function # for plotting degree 2 polynomial

linear_function(3, 9, (-5, 6), 1, annoInter=False)
''' plot y = 3x + 9
    data range for x: -5 to 5
    slope length: 1x
    don't plot intercept annotation ''' 
quadratic_function(2, 8, -16, (-20, 21), xInter=True, yInter=False, vertex=True, symLine=False)
''' plot y = 2x**2 + 8x - 16
    data range for x: -20 to 20
    plot roots (xInter) and vertex, but not y intercept nor line of symmetry '''

from linear_algebra import vector_properties # for plotting 2d vectors and returning their magnitude and direction
from linear_algebra import vector_addition # for plotting two 2d vectors and their sum before returning the latter
from linear_algebra import vector_multiplication # for multiplying with vectors and returning scalar, dot, or cross products
from linear_algebra import matrix_operations # for adding, subtracting, negating, transposing, multiplying, and dividing matrices
from linear_algebra import solve_system_of_equations # for using matrices to solve systems of equations
from linear_algebra import eigenvalues_eigenvectors # for calculating eigenvalues and eigenvectors of matrices

print(vector_properties(9, 9)) # get magnitude and amplitude of a 2x1 vector
print(vector_addition([9,7], [-1,-2])) # add two 2x1 vectors
print(vector_multiplication([2,3,1], [1,2,-2])) # multiply two 3x1 vectors
print(matrix_operations([[1,2,3],[4,5,6]], [[6,5,4],[3,2,1]], 'div')) # inversely multiply two 3x3 matrices
solve_system_of_equations([[4,2],[5,2]], [[14],[16]]) # 2x+4y=18; 6x+2y=34; find x and y
eigenvalues_eigenvectors([[2,0],[0,3]]) # get eigenvalues and eigenvectors of a 2x2 matrix


from integral_calculus import integral # for calculating an integral and plotting its area

integral() # no arguments because f(x) = 3x²+2x+1 and integral limits (0,3) are hardcoded in module
