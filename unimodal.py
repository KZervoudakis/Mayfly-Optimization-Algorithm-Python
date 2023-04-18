import numpy as np
# Sphere
def Sphere(pop,problem):
    x=pop['position']
    #pop['ProblemPosition']=-1*pop['position']
    #pop['cost']=sum((x**2))
    pop['cost']=np.sum(x**2)
    return pop
def SphereProblem(Dimensions):
    problem = {
            'CostFunction': Sphere, 
            'nVar': Dimensions, 
            'VarMin': -5*np.ones((Dimensions[0], Dimensions[1])),   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
            'VarMax':  5*np.ones((Dimensions[0], Dimensions[1])),    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
            'StopCriterion': None,
            'Max/Min': 1,
            'ProblemStructure': {'position': None, 'cost': None}#, 'ProblemPosition': None}
        }
    def f1(x) :
        return sum((x**2))
    from plotting import plot_3d, savefigures
    plt=plot_3d(f1, points_by_dim = 70, bounds = (-5, 5, -5, 5), cmap = 'twilight',plot_surface = True,plot_heatmap = True)
    savefigures(0,plt)
    
    return problem