# Project Title: A mayfly optimization algorithm (MA) in MATLAB
#
# Developers: K. Zervoudakis & S. Tsafarakis
#
# Contact Info: kzervoudakis@tuc.gr
#               School of Production Engineering and Management,
#               Technical University of Crete, Chania, Greece
#
# Researchers are allowed to use this code in their research projects, as
# long as they cite as:
# Zervoudakis, K., & Tsafarakis, S. (2020). A mayfly optimization algorithm.
# Computers & Industrial Engineering, 145, 106559.
# https://doi.org/10.1016/j.cie.2020.106559

import numpy as np
import time
import random
from operators import ContinousCrossover, ContinousMutation
from printingIter import printingperiter
# Particle Swarm Optimization
def MA(problem,IterPrint, MaxIter, MaxFuncEvals,curtrial,initialpop, mPopSize = 20, fPopSize=20, a1=1.0,a2=1.5, a3=1.5,beta=2,dance=5,fl=1,dance_damp=0.8,fl_damp=0.99,nc=20,gmax=0.8,gmin=0.8,gamma=0.4):
    # mPopSize = 20; fPopSize=20; a1=1.0;a2=1.5; a3=1.5;beta=2;dance=5;fl=1;dance_damp=0.8;fl_damp=0.99;nc=20;gmax=0.8;gmin=0.8;gamma=0.4
    namemethod='MA'
    VarMin = problem['VarMin']
    VarMax = problem['VarMax']
    nmm=round(0.05*(mPopSize)) # Number of Mutants
    nmf=round(0.05*(mPopSize)) # Number of Mutants         
    VelMax=0.1*(VarMax-VarMin);
    VelMin=-VelMax;
    g=gmax
    start_time = time.time()
    funcevals=-1;
    # Empty male Particle Template
    empty_maleparticle = {'velocity': None, 'best_position': None, 'best_cost': None}
    empty_maleparticle.update(problem['ProblemStructure'])
    # Empty female Particle Template
    empty_femaleparticle = {'velocity': None}
    empty_femaleparticle.update(problem['ProblemStructure'])
    # Extract Problem Info
    CostFunction = problem['CostFunction']
    nVar = problem['nVar']
    # Initialize Global Best
    gbest = problem['ProblemStructure'].copy(); gbest['cost'] = np.inf
    convergence = []
    # Create Initial male Population
    pop = []
    for i in range(0, mPopSize):
        pop.append(empty_maleparticle.copy())
        pop[i]['position'] = initialpop[i]['position'].copy()
        pop[i]['velocity'] = np.zeros((nVar[0], nVar[1]))
        pop[i]['cost'] = initialpop[i]['cost'].copy()
        funcevals+=1
        pop[i]['best_position'] = pop[i]['position'].copy()
        pop[i]['best_cost'] = pop[i]['cost'].copy()
        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy()
            gbest['cost'] = pop[i]['best_cost'].copy()
            if "ProblemPosition" in gbest:
                gbest['ProblemPosition'] = initialpop[i]['ProblemPosition'].copy()
        convergence.append(gbest['cost'])
    # Create Initial female Population
    popf = []
    for i in range(0, fPopSize):
        popf.append(empty_femaleparticle.copy())
        popf[i]['position'] = initialpop[i+mPopSize]['position'].copy()
        popf[i]['velocity'] = np.zeros((nVar[0], nVar[1]))
        popf[i]['cost'] = initialpop[i+mPopSize]['cost'].copy()
        funcevals+=1
        if popf[i]['cost'] < gbest['cost']:
            gbest['position'] = popf[i]['position'].copy()
            gbest['cost'] = popf[i]['cost'].copy()
            if "ProblemPosition" in gbest:
                gbest['ProblemPosition'] = initialpop[i+mPopSize]['ProblemPosition'].copy()
        convergence.append(gbest['cost'])
    # Mayfly Main Loop
    it=-1
    while (it<MaxIter and problem['StopCriterion'] == 'Iterations') or (funcevals<MaxFuncEvals-1 and problem['StopCriterion'] == 'Function Evaluations'):
        it+=1
        for i in range(0, fPopSize):
            # Update Females
            if popf[i]['cost'] > pop[i]['cost']:
                rmf=abs(pop[i]['position']-popf[i]['position'])
                popf[i]['velocity'] = g*popf[i]['velocity'] \
                    + a3*np.exp(-beta * (rmf ** 2))*(pop[i]['position'] - popf[i]['position'])
            else:
                popf[i]['velocity'] = g*popf[i]['velocity'] +fl*np.random.uniform(low=-1, high=1, size=(nVar[0], nVar[1]))
            # Apply Velocity Limits
            popf[i]['velocity'] = np.maximum(popf[i]['velocity'], VelMin)
            popf[i]['velocity'] = np.minimum(popf[i]['velocity'], VelMax)
            # Apply Position
            popf[i]['position'] += popf[i]['velocity']
            # Apply Position Limits
            popf[i]['position'] = np.maximum(popf[i]['position'], VarMin)
            popf[i]['position'] = np.minimum(popf[i]['position'], VarMax)
            popf[i] = CostFunction(popf[i],problem)
            funcevals+=1
            if popf[i]['cost'] < gbest['cost']:
                gbest['position'] = popf[i]['position'].copy()
                gbest['cost'] = popf[i]['cost'].copy()
                if "ProblemPosition" in gbest:
                    gbest['ProblemPosition'] = popf[i]['ProblemPosition'].copy()
            convergence.append(gbest['cost'])
            if funcevals>=MaxFuncEvals-1:
                break
        if funcevals>=MaxFuncEvals-1:
            break  
        for i in range(0, mPopSize):
            # Update Males
            if pop[i]['cost'] > gbest['cost']:
                rpbest=abs(pop[i]['best_position']-pop[i]['position'])
                rgbest=abs(gbest['position']-pop[i]['position'])
                pop[i]['velocity'] =g*pop[i]['velocity'] \
                    + a1*np.exp(-beta * (rpbest ** 2))*(pop[i]['best_position'] - pop[i]['position']) \
                    + a2*np.exp(-beta * (rgbest ** 2))*(gbest['position'] - pop[i]['position'])
            else:
                pop[i]['velocity'] =g*pop[i]['velocity']+dance*np.random.uniform(low=-1, high=1, size=(nVar[0], nVar[1]))
            # Apply Velocity Limits
            pop[i]['velocity'] = np.maximum(pop[i]['velocity'], VelMin)
            pop[i]['velocity'] = np.minimum(pop[i]['velocity'], VelMax)
            # Apply Position
            pop[i]['position'] += pop[i]['velocity']
            # Apply Position Limits
            pop[i]['position'] = np.maximum(pop[i]['position'], VarMin)
            pop[i]['position'] = np.minimum(pop[i]['position'], VarMax)
            pop[i] = CostFunction(pop[i],problem)
            funcevals+=1
            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost'].copy()
                if pop[i]['best_cost'] < gbest['cost']:
                    gbest['position'] = pop[i]['best_position'].copy()
                    gbest['cost'] = pop[i]['best_cost'].copy()
                    if "ProblemPosition" in gbest:
                        gbest['ProblemPosition'] = pop[i]['ProblemPosition'].copy()
            convergence.append(gbest['cost'])
            if funcevals>=MaxFuncEvals-1:
                break
        if funcevals>=MaxFuncEvals-1:
            break  
        # Sort Mayflies
        pop.sort(key=lambda x: x['cost'])
        popf.sort(key=lambda x: x['cost'])
        # Mate of mayflies
        for i in range(0, int(nc/2)):
            a = i #random.randint(0, mPopSize-1)
            b = i #random.randint(0, fPopSize-1)
            numberofpop=len(pop)
            numberofpopf=len(popf)
            pop.append(empty_maleparticle.copy())
            popf.append(empty_femaleparticle.copy())
            pop[numberofpop]['position'],popf[numberofpopf]['position']=ContinousCrossover(pop[a]['position'],popf[b]['position'],gamma)
            pop[numberofpop]['position'] = np.maximum(pop[numberofpop]['position'], VarMin)
            pop[numberofpop]['position'] = np.minimum(pop[numberofpop]['position'], VarMax)
            pop[numberofpop]['velocity'] = np.zeros((nVar[0], nVar[1]))
            pop[numberofpop]['best_position'] = pop[numberofpop]['position'].copy()
            popf[numberofpopf]['position'] = np.maximum(popf[numberofpopf]['position'], VarMin)
            popf[numberofpopf]['position'] = np.minimum(popf[numberofpopf]['position'], VarMax)
            popf[numberofpop]['velocity'] = np.zeros((nVar[0], nVar[1]))
            popf[numberofpop]['best_position'] = popf[numberofpop]['position'].copy()
            pop[numberofpop] = CostFunction(pop[numberofpop],problem)
            #print(pop[numberofpop]['cost'])
            pop[numberofpop]['best_cost'] = pop[numberofpop]['cost']#.copy()
            funcevals+=1
            if pop[numberofpop]['cost'] < gbest['cost']:
                gbest['position'] = pop[numberofpop]['position'].copy()
                gbest['cost'] = pop[numberofpop]['cost'].copy()
                if "ProblemPosition" in gbest:
                    gbest['ProblemPosition'] = pop[numberofpop]['ProblemPosition'].copy()
            convergence.append(gbest['cost'])
            if funcevals>=MaxFuncEvals-1:
                break
            popf[numberofpopf] = CostFunction(popf[numberofpopf],problem)
            funcevals+=1
            if popf[numberofpopf]['cost'] < gbest['cost']:
                gbest['position'] = popf[numberofpopf]['position'].copy()
                gbest['cost'] = popf[numberofpopf]['cost']#.copy()
                if "ProblemPosition" in gbest:
                    gbest['ProblemPosition'] = popf[numberofpopf]['ProblemPosition'].copy()
            convergence.append(gbest['cost'])          
            if funcevals>=MaxFuncEvals-1:
                break
        if funcevals>=MaxFuncEvals-1:
            break
        for i in range(0, int(nmm)):
            numberofpop=len(pop)
            pop.append(empty_maleparticle.copy())
            a = random.randint(0, mPopSize-1)
            pop[numberofpop]=pop[a].copy()
            pop[numberofpop]['position']=ContinousMutation(pop[a]['position'],problem)
            pop[numberofpop]['position'] = np.maximum(pop[numberofpop]['position'], VarMin)
            pop[numberofpop]['position'] = np.minimum(pop[numberofpop]['position'], VarMax)
            pop[numberofpop] = CostFunction(pop[numberofpop],problem)
            funcevals+=1
            if pop[numberofpop]['cost'] < pop[numberofpop]['best_cost']:
                pop[numberofpop]['best_position'] = pop[numberofpop]['position'].copy()
                pop[numberofpop]['best_cost'] = pop[numberofpop]['cost']#.copy()
                if pop[numberofpop]['best_cost'] < gbest['cost']:
                    gbest['position'] = pop[numberofpop]['best_position'].copy()
                    gbest['cost'] = pop[numberofpop]['best_cost'].copy()
                    if "ProblemPosition" in gbest:
                        gbest['ProblemPosition'] = pop[numberofpop]['ProblemPosition'].copy()
            convergence.append(gbest['cost'])
            if funcevals>=MaxFuncEvals-1:
                break  
        if funcevals>=MaxFuncEvals-1:
            break  
        for i in range(0, int(nmf)):
            numberofpop=len(popf)
            popf.append(empty_femaleparticle.copy())
            a = random.randint(0, fPopSize-1)
            popf[numberofpop]=popf[a].copy()
            popf[numberofpop]['position']=ContinousMutation(popf[a]['position'],problem)
            popf[numberofpop]['position'] = np.maximum(popf[numberofpop]['position'], VarMin)
            popf[numberofpop]['position'] = np.minimum(popf[numberofpop]['position'], VarMax)
            popf[numberofpop] = CostFunction(popf[numberofpop],problem)
            funcevals+=1
            if popf[numberofpop]['cost'] < gbest['cost']:
                gbest['position'] = popf[numberofpop]['position'].copy()
                gbest['cost'] = popf[numberofpop]['cost']#.copy()
                if "ProblemPosition" in gbest:
                    gbest['ProblemPosition'] = popf[numberofpop]['ProblemPosition'].copy()
            convergence.append(gbest['cost'])
            if funcevals>=MaxFuncEvals-1:
                break  
        if funcevals>=MaxFuncEvals-1:
            break    
        pop.sort(key=lambda x: x['cost'])
        del pop[mPopSize:]  
        popf.sort(key=lambda x: x['cost'])
        del popf[fPopSize:]
        g=gmax-((gmax-gmin)/MaxFuncEvals)*funcevals
        dance = dance*dance_damp
        fl = fl*fl_damp
        if it % IterPrint == 0:
            printingperiter(problem,it,gbest,namemethod,funcevals,curtrial)
    elapsed_time = time.time() - start_time
    if "ProblemPosition" in gbest:
        results = np.array([namemethod,gbest['ProblemPosition'], gbest['cost'], elapsed_time ,convergence, gbest['position']])
    else:
        results = np.array([namemethod,gbest['position'], gbest['cost'], elapsed_time ,convergence])    
    return results