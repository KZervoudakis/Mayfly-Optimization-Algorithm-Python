import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def MyPlotSO(CompMethod,trials,MaxFuncEvals,method,problem):
    ##% Plot
    figcount=0
    Convergence = np.zeros((int(CompMethod.shape[0]/trials), MaxFuncEvals))
    count=-1
    for i in range(0,int(CompMethod.shape[0]),trials):
        count+=1
        for j in range(trials):
            a=np.array(CompMethod[i+j,4])
            b=np.array(Convergence[count,:])
            Convergence[count,:] = a + b
    Convergence=problem['Max/Min']*Convergence
    Convergence= Convergence / trials   
    ##% Convergence curves
    PlotConvergence(Convergence,method)
    figcount+=1
    savefigures(figcount,plt)
    #savefigures(plots)
    ##% Violin Plots
    for i in range(2,4):
        #print(i)
        vec = CompMethod[:, i]
        matrix = vec.reshape((int(CompMethod.shape[0]/trials)),trials)
        matrix=matrix.T
        ViolinPlotting(matrix,trials,method)
        figcount+=1
        savefigures(figcount,plt)
    return

def PlotConvergence(Convergence,method):
    # Define colors, styles, and symbols for each line
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    #markers = ['o', 's', 'D', '*', '^']
    # Create a figure and axes
    fig, ax = plt.subplots()
    # Loop through each row of the matrix and plot it as a separate line
    for i in range(Convergence.shape[0]):
        ax.plot(Convergence[i], color=colors[i], linestyle=linestyles[i], label=method[i])
    # Add a legend to the plot
    ax.legend()
    ax.grid()
    # Add title and axis labels
    ax.set_title('Convergence characteristic curves')
    ax.set_xlabel('Function evaluations')
    ax.set_ylabel('Objective function value')
    # Display the plot
    #ax.show()
    return plt

def ViolinPlotting(matrix,trials,method):
    df = pd.DataFrame(matrix, columns=method)
    sns.set_context("talk", font_scale=1)
    plt.figure(figsize=(7,2*matrix.shape[1]))
    plt.grid()
    sns.violinplot(data=df, palette='pastel', bw=.5,orient="h")
    sns.stripplot(data=df,color="black", edgecolor="gray", orient="h")
    return plt

def plot_3d(func, points_by_dim = 50, title = '', bounds = None, cmap = 'twilight', plot_surface = True, plot_heatmap = True):
    from matplotlib.ticker import MaxNLocator, LinearLocator
    """
    Plots function surface and/or heatmap
    Parameters
    ----------
    func : class callable object
        Object which can be called as function.
    points_by_dim : int, optional
        points for each dimension of plotting (50x50, 100x100...). The default is 50.
    title : str, optional
        title of plot with LaTeX notation. The default is ''.
    bounds : tuple, optional
        space bounds with structure (xmin, xmax, ymin, ymax). The default is None.
    save_as : str/None, optional
        file path to save image (None if not needed). The default is None.
    cmap : str, optional
        color map of plot. The default is 'twilight'.
    plot_surface : boolean, optional
        plot 3D surface. The default is True.
    plot_heatmap : boolean, optional
        plot 2D heatmap. The default is True.
    """
    assert (plot_surface or plot_heatmap), "should be plotted at least surface or heatmap!"
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, points_by_dim)
    y = np.linspace(ymin, ymax, points_by_dim)
    a, b = np.meshgrid(x, y)
    data = np.empty((points_by_dim, points_by_dim))
    for i in range(points_by_dim):
        for j in range(points_by_dim):
            data[i,j] = func(np.array([x[i], y[j]]))
    a = a.T
    b = b.T
    l_a, r_a, l_b, r_b = xmin, xmax, ymin, ymax
    l_c, r_c = data.min(), data.max()
    levels = MaxNLocator(nbins=15).tick_values(l_c,r_c)
    if plot_heatmap and plot_surface:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2, projection='3d')
    else:
        fig = plt.figure()
        if plot_heatmap:
            ax1 = fig.gca()
        else:
            ax2 = fig.gca(projection='3d')
    #title = r"$\bf{" + title+ r"}$"
    #min_title = title[::]
    def base_plot():
        c = ax1.contourf(a, b, data , cmap=cmap, levels = levels, vmin=l_c, vmax=r_c)       
        name = title
        ax1.set_title( name, fontsize = 15)
        ax1.axis([l_a, r_a, l_b, r_b])
        fig.colorbar(c)
    if plot_surface:
        # Plot the surface.
        surf = ax2.plot_surface(a, b, data, cmap =  cmap,  linewidth=0, antialiased=False)
        ax2.contour(a, b, data, zdir='z', levels=30, offset=np.min(data), cmap=cmap)
        # Customize the z axis.
        ax2.set_xlabel('1st dim', fontsize=15)
        ax2.set_ylabel('2nd dim', fontsize=15)
        #ax2.set_zlabel('second dim', fontsize=10)
        ax2.set_zlim(l_c, r_c)
        ax2.zaxis.set_major_locator(LinearLocator(5))
        #ax2.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.tick_params(axis='z', pad=10)
        # Add a color bar which maps values to colors.
        if not plot_heatmap: fig.colorbar(surf)#, shrink=0.5, aspect=5)
        ax2.contour(a, b, data, zdir='z', offset=0, cmap =  cmap)
        ax2.view_init(30, 50)
        #ax2.set_title( min_title , fontsize = 15, loc = 'right')
    if plot_heatmap: base_plot()
    fig.tight_layout()
    #if save_as != None:
    #    plt.savefig(save_as, dpi = 900)
    plt.show()
    return fig

def savefigures(figcount,plt):
    # create a directory to store the figures
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/figure_{}.png'.format(figcount), dpi=900, bbox_inches="tight")
    return