'''
integral_calculus.py by Dominic Reichl, @domreichl
module for math_for_machine_learning.py that calculates an integral and plots its area
'''

def integral():
    print('-'*30 + '\nIntegral Calculus\n' + '-'*30)
    print('\nThe integral of a function is the area under it. This is the inverse of the function\'s derivative.')
    print('\nWhy is it relevant for machine learning?')
    print('Because a probability of some occurance might have to be computed between two limits.')
    print('\nLet\'s look at an example:')
    
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import integrate

    def f(x): return 3 * x**2 + 2 * x + 1 # function f
    print('\tFunction: 3x²+2x+1')
    
    x = range(0, 11) # array of x values from 0 to 10
    y = [f(a) for a in x] # get corresponding y values from f
    
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid() # set up plot
    plt.plot(x,y, color='purple') # plot x against f(x)
    
    section = np.arange(0, 3, 1/20) # integral area between limits of x 0 and 3
    print('\tIntegral area: 0 to 3')
    plt.fill_between(section,f(section), color='orange') # plot integral area

    i, e = integrate.quad(lambda x: f(x), 0, 3) # calculate approximate integral
    print('\tEstimated integral:', round(i))
    print('\tAbsolute error:', round(e, 10))

    plt.show() # display plot
    return i
