import numpy as np

def initialization(problem):    
    # Empty Solution Template
    empty_solution = problem['ProblemStructure']
    # Extract Problem Info
    CostFunction = problem['CostFunction']
    VarMin = problem['VarMin']
    VarMax = problem['VarMax']
    nVar = problem['nVar']
    # Create Initial Population
    initialpop = []
    for i in range(0, 200):
        #print(i)
        initialpop.append(empty_solution.copy())
        #initialpop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar)
        initialpop[i]['position'] = np.random.uniform(low=VarMin, high=VarMax, size=(nVar[0], nVar[1]))
        #pop=initialpop[i].copy()
        initialpop[i] = CostFunction(initialpop[i],problem)
        #if "ProblemPosition" in empty_solution:
        #    gbest['ProblemPosition'] = pop[i]['ProblemPosition'].copy()
    initialpop.sort(key=lambda x: x['cost'])    
    return initialpop