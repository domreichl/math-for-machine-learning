'''
linear_algebra.py by Dominic Reichl, @domreichl
module for math_for_machine_learning.py that introduces vectors and matrices

key concepts:
1.1. vectors: magnitude, direction, addition
1.2. vector multiplication: scalar, dot product, cross product
2.1. matrices: addition, subtraction, negation, transposition, multiplication, inverse
2.2. systems of equations
3.1. eigenvectors and eigenvalues
'''

'''
PART 1: VECTORS

A vector is a numberic element that has magnitude and direction.
Vectors describe spatial lines and planes; they enable calculations
that explore multi-dimensional relationships.
'''

def vector_properties(x, y):
    '''
        Takes the elements of a vector with 2 dimensions as input.
        Plots the vector and returns its magnitude and direction.
    '''

    print('-'*30 + '\nVector Properties\n' + '-'*30)

    import numpy as np # for building an array
    import matplotlib.pyplot as plt # for plotting the vector
    from math import sqrt, atan, degrees, log # for calculating vector properties

    v = np.array([x, y]) # define vector as a 2d array
    print('Vector:', v)

    # calculate vector magnitude
    vMag = sqrt(x**2 + y**2) # raw equation
    vMagAlt = np.linalg.norm(v) # numpy equation
    vMag == vMagAlt # True
    print('Magnitude:', vMag)

    # calculate vector direction (vector angle as inverse tangent)
    if x > 0: vAtan = degrees(atan(y/x))
    elif x < 0 and y > 0: vAtan = degrees(atan(y/x)) + 180
    elif x < 0 and y < 0: vAtan = degrees(atan(y/x)) - 180
    vAtanAlt = np.degrees(np.arctan2(y, x)) # numpy equation (saves 2 lines of code)
    vAtan == vAtanAlt # True
    print('Direction:', vAtan)

    # plot vector as arrow in a coordinate system (quiver plot)
    plt.grid()
    plt.quiver([0],[0], *v, angles='xy', scale_units='xy', scale=1, color='green')
    plt.xlim(-10, 10); plt.ylim(-10, 10)
    plt.show()
    
    return vMag, vAtan

def vector_addition(v1, v2):
    '''
        Takes as input two 2d vectors as lists.
        Plots the vectors and their sum, then returns the latter.
    '''

    print('-'*30 + '\nVector Addition\n' + '-'*30)

    import numpy as np # for building arrays
    import matplotlib.pyplot as plt # for plotting vectors

    v1 = np.array(v1) # first vector as array
    v2 = np.array(v2) # second vector as array
    vSum = v1 + v2 # vector addition
    vectors = np.array([v1, v2, vSum])
    print('Vector 1:', v1)
    print('Vector 2:', v2)
    print('Vector sum:', vSum)

    # plot vectors as arrows (quiver plot)
    plt.grid()
    plt.quiver([0],[0], vectors[:,0], vectors[:,1], angles='xy', scale_units='xy', scale=1, color=['g', 'b', 'r'])
    plt.xlim(-10, 10); plt.ylim(-10, 10)
    plt.show()

    return vSum

def vector_multiplication(x, y):
    '''
        Takes as input 2d or 3d vector(s) as list(s) and/or number(s) as integer(s).
        Returns scalar, dot, or cross product as output.
    '''

    print('-'*30 + '\nVector Multiplication\n' + '-'*30)

    import numpy as np
    
    if type(x) == list and type(y) == int:
        product = np.array(x) * y # scalar product
        print('Vector:', np.array(x))
        print('Multiplier:', y)
        print('Scalar product:', product)
        return product
    elif type(x) == int and type(y) == list:
        product = x * np.array(y) # scalar product
        print('Multiplier:', y)
        print('Vector:', np.array(y))
        print('Scalar product:', product)
        return product
    elif type(x) == list and type(y) == list:
        v1 = np.array(x)
        v2 = np.array(y)
        print('Vector 1:', v1)
        print('Vector 2:', v2)
        if len(x) == 2:
            product = v1 @ v2 # dot product; alternative: np.dot(v1,v2)
            print('Dot product:', product)
            return product
        elif len(x) == 3:
            product = np.cross(v1,v2) # cross product
            print('Cross product:', product)
            return product

'''
PART 2: MATRICES

A matrix is an array of numbers that are arranged in rows and columns.
For example, a 2x3 matrix has two rows and three columns.
'''

def matrix_operations(A, B, operation='add'):
    '''
        Takes as input 2 matrices as nested lists and an operation.
        Possible operations:
            - addition ('add'; default)
            - subtraction ('sub')
            - negation ('neg')
            - transposition ('t')
            - multiplication ('mul')
            - division ('div'; inverse multiplication)
        Returns result of operation.
    '''

    print('-'*30 + '\nMatrix Operation\n' + '-'*30)

    import numpy as np

    A = np.matrix(A) # .matrix allows multiplication with * and inverse with .I
    B = np.matrix(B)

    if operation == 'add': return print(A+B) # sum of matrices
    if operation == 'sub': return print(A-B) # difference of matrices
    if operation == 'neg': return print(-A,-B) # negation of matrices
    if operation == 'transpose': return print(A.T,B.T) # transposition of matrices
    if operation == 'mul': # matrix multiplication if allowed by shapes
        if np.shape(A)[0] == np.shape(B)[1] and np.shape(A)[1] == np.shape(B)[0]:
            return print(A*B) # dot product; alternatives: A@B; np.dot(A,B)
        else:
            return print('These matrices cannot be multiplied.')
    if operation == 'div': # matrix inverse multiplication if allowed by shapes
        if np.shape(A)[0] == np.shape(B)[0] and np.shape(A)[1] == np.shape(B)[1]:
            return print(A*B.I) # dot product; alternative: A@np.linalg.inv(B)
        else:
            return print('These matrices cannot be divided.')

def solve_system_of_equations(xy, results):
    '''
        Takes as input 2 matrices as nested lists representing equations.
        For example, 'xy' might be [[2,4],[6,2]] for equations
        2x+4y and 6x+2y, and 'results' might be [[18,34]].
        Returns values for x and y that solve the equation system.
    '''

    print('\n' + '-'*30 + '\nSolving System of Equations\n' + '-'*30)

    import numpy as np

    xy = np.matrix(xy)
    results = np.matrix(results)
    
    XY = xy.I @ results # inverse multiplication
    x = float(XY[0][0])
    y = float(XY[1][0])
    
    print('x is', x, '\ny is', y)
    return x,y # return values for x and y as floats in a tuple

'''
PART 3: EIGENVECTORS AND EIGENVALUES

An eigenvalue is a scalar multiplier that produces an eigenvector
via linear transformation of a non-zero vector.
'''

def eigenvalues_eigenvectors(A):
    '''
        Takes as input a matrix as a nested list.
        Returns eigenvalues and eigenvectors.
    '''

    print('\n' + '-'*30 + '\nEigenvectors, Eigenvalues\n' + '-'*30)

    import numpy as np

    A = np.array(A) # turn list into array
    eVals, eVecs = np.linalg.eig(A) # get eigenvalues and eigenvectors

    print('Matrix:\n', A)
    print('Lambdas:', eVals[0], 'and', eVals[1])
    print('Vectors:', eVecs[:,0], 'and', eVecs[:,1])

    return eVals, eVecs
