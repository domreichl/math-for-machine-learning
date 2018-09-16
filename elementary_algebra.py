'''
elementary_algebra.py
    by Dominic Reichl, @domreichl
module for math_for_machine_learning.py that plots polynomials
'''

# degree 1 polynomial: y = mx + c
def linear_function(m, c, r=(-10, 11), s=0, annoInter=True):
    '''
        Creates and plots a data frame for a linear equation,
            optionally with annotations for x and y intercepts.
        m: slope
        c: constant
        r: range for x column of data frame as tuple; default is -10 to 10
        s: slope length for plot; default is 0x
        annoInter: annotate x and y intercepts; default is True
    '''

    import pandas as pd # for data frames
    from matplotlib import pyplot as plt # for graphs

    df = pd.DataFrame({'x': range(r[0], r[1])}) # create data frame with x column
    df['y'] = m*df['x'] + c # add y column by applying linear equation
    print('Linear function\ny = %dx + %d\n' %(m,c), df) # display data frame

    plt.plot(df.x, df.y, color='grey', marker = 'o') # create plot
    plt.xlabel('x'); plt.ylabel('y') # label axes
    plt.grid(); plt.axhline(); plt.axvline() # add grid and axis lines

    if annoInter == True:
        plt.annotate('x intercept', (-c/m, 0)) # annotate x intercept
        plt.annotate('y intercept', (0, c)) # annotate y intercept

    mx = [0, s]; my = [0*m+c, s*m+c] # set slope for sx
    plt.plot(mx, my, color='red', lw=3) # plot slope line

    plt.show() # display plot
    return

# degree 2 polynomial: y = ax² + bx + c
def quadratic_function(a, b, c, r=(-10, 11), xInter=True, yInter=True, vertex=True, symLine=True):
    '''
        Creates and plots a data frame for a quadratic equation,
            optionally with parabola intercepts, a vertex, a and symmetry line.
        a, b: coefficients
        c: constant
        r: range for x column of data frame as tuple; default is -10 to 10
        xInter: plot x intercepts (polynomial roots); default is True
        yInter: plot y intercept; default is True
        vertex: plot vertex (max/min of curve); default is True
        symLine: plot line of symmetry; default is True
    '''

    import pandas as pd # for data frames
    from matplotlib import pyplot as plt # for graphs
    from math import sqrt # for calculating polynomial roots
    
    df = pd.DataFrame ({'x': range(r[0], r[1])}) # create data frame with x column
    df['y'] = a*df['x']**2 + b*df['x'] + c # add y column by applying quadratic equation
    print('Quadratic function\ny = %dx² + %dx + %d\n' %(a,b,c), df) # display data frame

    plt.plot(df.x, df.y, color='grey') # create plot
    plt.xlabel('x'); plt.ylabel('y') # label axes
    plt.grid(); plt.axhline(); plt.axvline() # add coordinate lines

    vx = (-1*b) / (2*a) # get x at line of symmetry
    vy = a*vx**2 + b*vx + c # get y at line of symmetry
    miny = df.y.min(); maxy = df.y.max() # get min & max y values from data frame
    sx = [vx, vx]; sy = [miny, maxy] # get symmetry line

    if b**2-4*a*c > 0: # check if discriminant is positive
        x1 = (-b-sqrt((b**2)-(4*(a*c)))) / (2*a) # calculate first polynomial root
        x2 = (-b+sqrt((b**2)-(4*(a*c)))) / (2*a) # calculate second polynomial root
        print('Roots:', round(x1, 2), 'and', round(x2, 2))
        if xInter == True: plt.scatter([x1, x2], [0, 0], color='black') # plot x intercepts
    elif b**2-4*a*c == 0: # check if polnynomial has one root
        print('Root is vertex:', (vx, vy))
    else: # polnynomial has no roots
        print('No roots.')
        
    if yInter == True: plt.scatter(0, c, color='black') # plot y intercept
    if vertex == True: plt.scatter(vx, vy, color='magenta') # plot vertex
    if symLine == True: plt.plot(sx, sy, color='magenta') # plot symmetry line

    plt.show() # display plot
    return
