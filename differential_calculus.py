'''
differential_calculus.py by Dominic Reichl, @domreichl
module for math_for_machine_learning.py that introduces differential calculus

With differential calculus, we use derivatives to find the slope of a curve at a given point x,
i.e., at an infitesimally small line between x1 and x2 (infitesimally small delta x).
A derivative is the limit of a function at a point x as delta x approaches 0).
With derivatives, we can find the local minimum/maximum of a function.
'''

def average_rate_of_change():
    print('-'*30 + '\nAverage Rate of Change\n' + '-'*30)
    print('\nFunction f(x) = x²+x calculates the distance travelled for a given number of seconds.')
    print('\nAs you can see in the green curve, since the function is quadratic, the the rate of change is not constant, but increasing; the represented phenomenon is accelerating.')
    print('\nNote that acceleration measures change in velocity over time, and velocity measures change in distance over time.')
    print('\nThis complexity requires differential calculus.')
    print('\nBut first we need a SECANT LINE, which is a straight line (here in pink) between two points of a function (here x=2 and x=7).')
    print('\nThe average rate of change (= SLOPE) for a part of a function is delta y/delta x, whereby delta y is the change in y over the secant line, i.e., y2-y1, and delta x is the change in x, i.e., x2-x1.')

    import numpy as np
    from matplotlib import pyplot as plt
    
    def f(x): return x**2 + x
    
    x = np.array(range(0, 11)) # create array of x values from 0 to 10
    s = np.array([2,7]) # create array for secant line

    #calculate rate of change
    x1 = s[0]
    x2 = s[-1]
    y1 = f(x1)
    y2 = f(x2)
    a = (y2 - y1)/(x2 - x1)
    print('\nHere we have (f(7)-f(2)/(7-2) = (7²+7-2²+2)/5, which gives us an average velocity of ' + str(a) + 'm/s.\n')
    
    # set up graph
    plt.xlabel('Seconds')
    plt.ylabel('Meters')
    plt.title('f(x) = x²+x')
    plt.grid()

    plt.plot(x,f(x), color='green') # plot x against f(x)
    plt.plot(s,f(s), color='magenta') # plot secant line

    plt.annotate('Average Velocity =' + str(a) + ' m/s',((x2+x1)/2, (y2+y1)/2))
    plt.show()
    
    return a

def limits():
    print('-'*30 + '\nLimits\n' + '-'*30)
    print('\nThe purpose of differential calculus is to use derivatives to get the slope (average rate of change) of a single point on a function (i.e., when the secant line is infinitesimally small).')
    print('\nBut before we use derivatives, let\'s look at how a secant line gets smaller; in particular, how a function\'s value changes as delta x decreases, i.e., as x2 approaches x1.')

    from matplotlib import pyplot as plt

    def f(x): return x**2 + x

    # create array of x values from 0 to 10 to plot
    x = list(range(0,5))
    x.append(4.25)
    x.append(4.5)
    x.append(4.75)
    x.append(5)
    x.append(5.25)
    x.append(5.5)
    x.append(5.75)
    x = x + list(range(6,11))

    # get corresponding y values from function
    y = [f(i) for i in x]

    # set up graph
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = x²+x')
    plt.grid()

    # plot function
    plt.plot(x,y, color='lightgrey', marker='o', markeredgecolor='green', markerfacecolor='green')

    # plot f(x) when x = 5
    zx = 5
    zy = f(zx)
    plt.plot(zx, zy, color='red', marker='o', markersize=10)
    plt.annotate('x=' + str(zx),(zx, zy), xytext=(zx - 0.5, zy + 5))

    # plot f(x) when x = 5.1
    posx = 5.25
    posy = f(posx)
    plt.plot(posx, posy, color='blue', marker='<', markersize=10)
    plt.annotate('x=' + str(posx),(posx, posy), xytext=(posx + 0.5, posy - 1))

    # plot f(x) when x = 4.9
    negx = 4.75
    negy = f(negx)
    plt.plot(negx, negy, color='orange', marker='>', markersize=10)
    plt.annotate('x=' + str(negx),(negx, negy), xytext=(negx - 1.5, negy - 1))

    print('\nIn this plot, you can see the points narrowing in on f(5).')
    print('\nIf you imagine the points coming as closely together as possible without actually being x=5, you get a LIMIT, namely, lim f(x) as x approaches 5.\n')
    plt.show()    
    return

def discontinuity():
    print('-'*30 + '\nDiscontinuity\n' + '-'*30)
    print('\nThe graph you see now has a gap at x=0, which means that the function is NON-CONTINUOUS at that point.')
    print('\nSince x=0 is not part of the function\'s domain, we can\'t evaluate f(0).')
    print('\nBut what we can do is calculate lim f(0) as x approaches 0.\n')

    from matplotlib import pyplot as plt

    def f(x):
        if x != 0:
            return -(12/(2*x))**2

    # create array of x values
    x = range(-20, 21)

    # get corresponding y values from function
    y = [f(a) for a in x]

    # set up the graph
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('y=-(12/2x)², x!=0')
    plt.grid()

    # plot x against f(x)
    plt.plot(x,y, color='green')

    # plot a circle at the gap
    xy = (0,f(1))
    plt.annotate('O',xy, xytext=(-0.7, -37),fontsize=14,color='green')

    # plot f(x) when x = 1
    posx = 1
    posy = f(posx)
    plt.plot(posx, posy, color='blue', marker='<', markersize=5)

    # plot f(x) when x = -1
    negx = -1
    negy = f(negx)
    plt.plot(negx, negy, color='orange', marker='>', markersize=5)

    plt.show()
    return

def differentiation():
    print('-'*30 + '\nDifferentiation\n' + '-'*30)
    print('\nSo far we have seen what secant lines and limits look like, but remember that our goal is to find the slope of a single point of a function.')
    print('\nWe get that slope by calculating a delta for x1 and x2 values that are infinitesimally close together.')
    print('\nLet\'s say we want to know the slope at x=5.')
    print('\nWhat we have to do is define a second point that is infinitesimally close to 5, e.g., 5.000000001, and approximate the tangent slope (here in pink) for the distance between them (infinitesimally small delta x).')

    def f(x): return x**2 + x

    from matplotlib import pyplot as plt

    # create array of x values from 0 to 10
    x = list(range(0, 11))

    # use function to get the y values
    y = [f(i) for i in x]

    # set x1 point
    x1 = 5
    y1 = f(x1)

    # set the x2 point very close to x1
    x2 = 5.000000001
    y2 = f(x2)

    # set up the graph
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x)=x²+x')
    plt.grid()

    # plot the function
    plt.plot(x,y, color='green')

    # plot point x1
    plt.scatter(x1,y1, c='red')
    plt.annotate('x=' + str(x1),(x1,y1), xytext=(x1-0.5, y1+3))

    # approximate the tangent slope and plot it
    m = (y2-y1)/(x2-x1)
    xMin = x1 - 3
    yMin = y1 - (3*m)
    xMax = x1 + 3
    yMax = y1 + (3*m)
    plt.plot([xMin,xMax],[yMin,yMax], color='magenta')
    print('\nSo we calculate delta y/delta x = (y2-y1)/(x2-x1) = (f(5.000000001)-f(5))/(5.000000001-5) and get the slope of our function at point x=5: approximately ' + str(m) + '.')
    print('\nTo be more precise, we can use limits.')
    print('\nSetting h as delta x, we can calculate lim (f(x+h)-f(x))/h as h approaches 0. This is our first DERIVATIVE of f(x): f\'(x) or d/dx f(x).')
    print('\nLagrange notation: f\'(x) = lim (f(x+h)-f(x))/h as h approaches 0.')
    print('\nLeibniz notation: d/dx f(x) = lim (f(x+h)-f(x))/h as h approaches 0.')
    print('\nA simpler way to calculate derivatives is by applying derivative rules:')
    print('\t - constants become zeros')
    print('\t - x**n becomes nx**n-1 (power rule)')
    print('In our example: f(x) = x² + x -> f\'(x) = 2x**1 + 1x**0 = 2x + 1; if x=5 -> f\'(x) = 2*5+1 = 11')
    print('\nNote that not any function is differentiable at a given point. There are three requirements: 1. the function must be continuous at that point; 2. the tangent line at that point must not be vertical; 3. the line must be smooth at that point (i.e., no sudden change of direction).\n')

    plt.show()
    return m

def critical_points():
    print('-'*30 + '\nCritical Points\n' + '-'*30)
    print('\nDerivatives give us interesting information about a function.')
    print('\nIn the graph here, you can see a function (green), its derivative (purple), and three slopes (red tangents).')
    print('\nThe tangent line on top is perfectly horizontal because the slope at point x=5 is 0.')
    print('\nAccordingly, the derivative line, which represents the function\'s slopes, crosses 0 at point x=5.')
    print('\nWhenever the derivative of a function is zero, a CRITICAL POINT is reached, at which the function line changes its direction.')
    print('\nA critical point may represent a local maximum, a local minimum, or an inflexion point.')
    print('\nWe can find minima/maxima by calculating the second order derivative.')
    print("\nIf we have k(x)=-10x²+100x+3, its derivative k'(x) is -20x+100, and its second-order derivative k'' is -20.")
    print("\nBecause k'' is negative, k'(5) is a local maximumm; if it were positive, k'(5) would be a maximum.")

    from matplotlib import pyplot as plt
    
    def k(x): return -10*(x**2) + (100*x)  + 3 # function k(x) = -10x²+100x+3
    def kd(x): return -20*x + 100 # derivative k'(x) = -20x+100
    
    x = list(range(0, 11)) # create an array of x values
    y = [k(i) for i in x] # use function to get the y values
    yd = [kd(i) for i in x] # use derivative function to get derivative values

    # set up the graph
    plt.xlabel('x (time in seconds)')
    plt.ylabel('k(x) (height in feet)')
    plt.title('k(x) = -10x²+100x+3')
    plt.xticks(range(0,15, 1))
    plt.yticks(range(-200, 500, 20))
    plt.grid()

    plt.plot(x,y, color='green') # plot the function
    plt.plot(x,yd, color='purple') # # plot the derivative

    # plot tangent slopes for x = 2, 5, 8
    x1 = 2
    x2 = 5
    x3 = 8
    plt.plot([x1-1,x1+1],[k(x1)-(kd(x1)),k(x1)+(kd(x1))], color='red')
    plt.plot([x2-1,x2+1],[k(x2)-(kd(x2)),k(x2)+(kd(x2))], color='red')
    plt.plot([x3-1,x3+1],[k(x3)-(kd(x3)),k(x3)+(kd(x3))], color='red')

    plt.show()

    print('\n' + '-'*50 + '\nHowever, if the slope of a function flattens out to 0 on a number of points (here forming a \'saddle\'), we have multiple critical points, which are not minima or maxima. How do we know?')
    print('\n1. Our function is x³-6x²+12x+2 (green).')
    print('2. Its second order derivative is 6x-12 (pink), which is not a horizontal line.')
    print('3. For x=2, the second derivatve is 6*2-12 = 0, so neither negative nor positive, so neither a max nor a min.')
    print('\nIn machine learning, finding maxima and minima is useful for optimizing a function for a specific variable.')
    print('\nFor example, you want the cost of a product to be neither too high nor too low, but at a critical point that represents a maximum (of sales). So you optimize your sales function for the price variable x by finding the value of x where the first derivative is zero and the second derivative is negative.\n')

    def v(x): return (x**3) - (6*(x**2)) + (12*x) + 2
    def vd(x): return (3*(x**2)) - (12*x) + 12
    def v2d(x): return (3*(2*x)) - 12 # second order derivative

    x = list(range(-5, 11))
    y = [v(i) for i in x]
    yd = [vd(i) for i in x]
    y2d = [v2d(i) for i in x] # get values for second order derivative

    plt.xlabel('x')
    plt.ylabel('v(x)')
    plt.title('v(x)=x³-6x²+12x+2')
    plt.xticks(range(-10,15, 1))
    plt.yticks(range(-2000, 2000, 50))
    plt.grid()

    plt.plot(x,y, color='green')
    plt.plot(x,yd, color='purple')
    plt.plot(x,y2d, color='magenta') # plot second order derivative

    plt.show()
    return

def partial_derivatives():
    print('-'*30 + '\nPartial Derivatives\n' + '-'*30)
    print('\nIf we have multivariate functions, we need to calculate PARTIAL DERIVATIVES, with which we can compute the rate of change of a function of many variables with respect to one of those variables.')
    print('\nFor example, the partial derivative of f(x,y) = x²+y² is ∂f(x,y)/∂x = ∂(x²+y²)/∂x.')
    print('\nThe partial derivative of f(x,y) with respect to x is ∂f(x,y)/∂x = 2x+0 = 2x')
    print('\nThe partial derivative of f(x,y) with respect to y is ∂f(x,y)/∂y = 0+2y = 2y')
    print('\nWith this information, we can find GRADIENTS, i.e., slopes (arrows in graph) for multi-dimensional surfaces (circles in graph).')
    print('\nIn our example: gradient grad(f(x,y)) = [[2x],[2y]], i.e., a 2-dimensional vector.')
    print('\nThe arrows in the graph show the gradient direction, and their width indicate the gadient value. As you can see, the gradient decreases as the function approaches a minimum. Note also that the gradient is always perpendicular to the colored contures.')
    print('\nIn machine learning, gradients are used for the gradient descent algorithm, where you take small steps from some starting guess in the gradient direction until the gradient is close to zero.\n')

    import matplotlib.pyplot as plt
    import numpy as np
    import math

    # create a uniform grid
    el = np.arange(-5,6)
    nx, ny = np.meshgrid(el, el, sparse=False, indexing='ij')

    # flatten the grid to 1 dimension and compute the value of the function z
    x_coord = []
    y_coord = []
    z = []
    for i in range(11):  
        for j in range(11):
            x_coord.append(float(-nx[i,j]))
            y_coord.append(float(-ny[i,j]))       
            z.append(nx[i,j]**2 + ny[i,j]**2)

    # vector arithmetic to get the x and y gradients        
    x_grad = [-2 * x for x in x_coord]
    y_grad = [-2 * y for y in y_coord]

    # plot the arrows using width for gradient
    plt.xlim(-5.5,5.5)
    plt.ylim(-5.5,5.5)
    for x, y, xg, yg in zip(list(x_coord), list(y_coord), list(x_grad), list(y_grad)):
        if x != 0.0 or y != 0.0: # avoid zero divide when scaling the arrow
            l = math.sqrt(xg**2 + yg**2)/2.0 # arrow width proporitional to gradient value
            plt.quiver(x, y, xg, yg, width = l, units = 'dots')

    # plot the countours of the function surface
    z = np.array(z).reshape(11,11)    
    plt.contour(el, el, z)
    
    plt.show()
    return
