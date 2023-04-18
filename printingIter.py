def printingperiter(problem,it,gbest,namemethod,funcevals,curtrial):

    print('Problem: {}, Method: {}, Trial: {}, Iteration: {}, Function Evaluations: {}, Best Cost = {}'.format(problem['CostFunction'].__name__,namemethod,curtrial+1,it, funcevals, gbest['cost']))
    return