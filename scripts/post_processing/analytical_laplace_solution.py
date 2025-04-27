import sys
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation

matplotlib.use('Qt5Agg')

def analytical_solution(x, y, U0=1, n_terms=50):
    """Analytical solution to Laplace equation with given boundary conditions"""
    
    sol = 0.0
    
    for n in range(1, 2*n_terms, 2):  # Only odd terms
        term = (1.0/(n*np.pi*np.sinh(n*np.pi))) * np.sin(n*np.pi*x) * np.sinh(n*np.pi*(1-y))
        sol += term
    
    sol *= 4.0*U0

    return sol

def analytical_laplace_solution(x,y,domain,bc,nterms):
    '''
    function to analytically determine the solution of the Laplace equation in 2D
    when using Dirichlet BCs.
    '''

    # Set domain bounds
    xs = domain[0]  # x-coordinate for start of domain
    xe = domain[1]  # x-coordinate for end of domain
    ys = domain[2]  # y-coordinate for start of domain
    ye = domain[3]  # y-coordinate for end of domain

    # boundary values
    bc1 = bc[0] # bc at start of x
    bc2 = bc[1] # bc at end of x
    bc3 = bc[2] # bc at start of y
    bc4 = bc[3] # bc at end of y

    # get number of cells in domain
    ncells = len(x)

    # initialize solution vector for each cell
    laplace_solution = np.zeros(ncells)

    if (ncells != len(y)):
       print("[E] Sizes of x and y coordiante arrays are not the same. Code exitting!", flush=True)
       sys.exit(12)

    # loop over all cells and compute analytical solution
    for icell in range(ncells):
        xloc = x[icell]
        yloc = y[icell]

        laplace_solution[icell] = analytical_solution(xloc,yloc,bc[2],nterms)

    return laplace_solution

def read_laplace_solution(filename):

    laplace_data = np.loadtxt(filename,delimiter=",")
    x            = laplace_data[:,0]
    y            = laplace_data[:,1]
    sol          = laplace_data[:,2]

    return x,y,sol

def compute_error(sol,val):

    ncells = len(sol)
    error  = np.zeros(ncells)
    max_error_index = -1

    max_error = -1.0e30

    for icell in range(ncells):
        error[icell] = np.abs(sol[icell] - val[icell])
        if (error[icell] > max_error):
            max_error = error[icell]
            max_error_index = icell

    return error, max_error, max_error_index


def compute_error_norms(error_vec):

    l2_norm = 0.0
    linf_norm = -1.0e30

    ncells = len(error_vec)

    for icell in range(ncells):
        l2_norm += error_vec[icell]*error_vec[icell]
        linf_norm = max(linf_norm,np.abs(error_vec[icell]))

    l2_norm   = np.sqrt(l2_norm)/float(ncells)
    linf_norm = linf_norm/float(ncells)

    return l2_norm, linf_norm

def compute_order_of_accuracy(dx,l2,linf):

    l2_order   = 0.0
    linf_order = 0.0

    nruns = len(linf)
    l2_orders   = np.zeros(nruns-1)
    linf_orders = np.zeros(nruns-1)

    for irun in range(1,nruns):
        l2_orders[irun-1]   = np.log(l2[irun]/l2[irun-1])/np.log(dx[irun]/dx[irun-1])
        linf_orders[irun-1] = np.log(linf[irun]/linf[irun-1])/np.log(dx[irun]/dx[irun-1])

    return l2_orders, linf_orders

def save_contour_plot(contour_data,nx,out_dir,bc_val):

    # Perform data sorting
    # This is needed for multi-process runs as the data will not be sorted based on the
    # physical coordinate but based on the sequence of processes
    sorted_indices = np.lexsort((contour_data[:,1],contour_data[:,0]))
    sorted_data    = contour_data[sorted_indices]

    # Get individual contour variables and perform reshape
    X = sorted_data[:,0].reshape(nx,nx)
    Y = sorted_data[:,1].reshape(nx,nx)
    S = sorted_data[:,2].reshape(nx,nx)
    E = sorted_data[:,3].reshape(nx,nx)
    A = sorted_data[:,4].reshape(nx,nx)

    # set colorbar limits
    min_lim = 0
    max_lim = bc_val

    # set contour skip
    skip = nx//5

    # set fontsizes
    subplot_title_fontsize=13
    axes_label_fontsize=12

    # Create subplot structure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 10), constrained_layout=True)

    # solution subplot
    levels = np.linspace(min_lim,max_lim, nx+1)
    cont1 = ax1.contourf(X, Y, S, levels=levels, cmap='rainbow', extend='both')
    fig.colorbar(cont1, ax=ax1, fraction=0.04, pad=0.02, aspect=10, orientation='horizontal', ticks=levels[::skip])
    ax1.set_title(r"Numerical Solution", fontsize=subplot_title_fontsize)
    ax1.set_xlabel('X', fontsize=axes_label_fontsize)
    ax1.set_ylabel('Y', fontsize=axes_label_fontsize)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.set_aspect('equal')  

    # analytical subplot
    cont2 = ax2.contourf(X, Y, A, levels=levels, cmap='rainbow', extend='both')
    fig.colorbar(cont2, ax=ax2, fraction=0.04, pad=0.02, aspect=10, orientation='horizontal', ticks=levels[::skip])
    ax2.set_title(r"Analytical Solution", fontsize=subplot_title_fontsize)
    ax2.set_xlabel('X', fontsize=axes_label_fontsize)
    ax2.set_ylabel('Y', fontsize=axes_label_fontsize)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_aspect('equal')  

    # error subplot (plotted on log scale)
    log_max = 10.0
    log_min = 1.0e-7
    nlevels = int(np.log10(log_max/log_min)) + 1

    levels = np.logspace(np.log10(log_min),np.log10(log_max), nx + 1)
    cont3 = ax3.contourf(X, Y, E, levels=levels, extend='both', cmap='rainbow', norm=LogNorm())
    cbar3 = fig.colorbar(cont3, ax=ax3, fraction=0.04, pad=0.02, aspect=10, orientation='horizontal')
    
    tick_values = np.logspace(np.log10(log_min), np.log10(log_max), num=nlevels)
    cbar3.set_ticks(tick_values[::2])
    cbar3.ax.xaxis.set_major_formatter(LogFormatterSciNotation())  

    ax3.set_title(r"Absolute Error, $\varepsilon$", fontsize=subplot_title_fontsize)
    ax3.set_xlabel('X', fontsize=axes_label_fontsize)
    ax3.set_ylabel('Y', fontsize=axes_label_fontsize)
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,1)
    ax3.set_aspect('equal')  

    # Maximize plot window
    plt.get_current_fig_manager().window.showMaximized()

    # name of output file
    output_filename = "{:s}/laplace_solution_error_nx_{:d}.png".format(out_dir,nx)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')

    # Close plot window
    plt.close(fig)

    return None

def output_error_to_file(x,y,error,nx,out_dir):
    
    out_data = np.column_stack((x,y,error))
    outfilename = "{:s}/error_nx_{:d}.dat".format(out_dir,nx)
    np.savetxt(outfilename,out_data)

def plot_error_dx(dx,l2_err,linf_err,out_dir):
    
    # Set fontsizes
    axes_fontsize   = 20
    legend_fontsize = 20
    tick_fontsize   = 15
    tick_length     = 8
    tick_width      = 1

    dx_min = min(np.min(dx),1.0e-2)
    dx_max = max(np.max(dx), 0.3)

    plt.figure(2)
    plt.loglog(dx,l2_err  ,'r-D',label=r"$p = 2$"  )
    plt.loglog(dx,linf_err,'b-D',label=r"$p = \infty$")
 
    plt.title(r"Error ($\varepsilon$) vs Grid Spacing ($\Delta x$)", fontsize=20)
    
    # X axis properties
    plt.xlabel(r"$\Delta x$",fontsize=axes_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.xlim(dx_min,dx_max)
    
    # y axis properties
    plt.ylabel(r"$||\varepsilon||_p$",fontsize=axes_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    #plt.ylim(1e-4,1e-1)

    # Increase tick size for increased readability
    plt.tick_params(which='major',
                    axis='both',
                    length=tick_length,
                    width=tick_width)
    
    plt.tick_params(which='minor',
                    axis='both',
                    length=6,
                    width=tick_width)
    
    plt.legend(frameon=False,fontsize=legend_fontsize)

    plt.annotate('', xy=(dx[1], linf_err[1]), xytext=(dx[1], linf_err[2]),
                    arrowprops=dict(arrowstyle='-', color='black',
                    linewidth=1.5, linestyle=(0,(4,3))))
    
    plt.annotate('', xy=(dx[2], linf_err[2]), xytext=(dx[1], linf_err[2]),
                    arrowprops=dict(arrowstyle='-', color='black',
                    linewidth=1.5, linestyle=(0,(4,3))))
    
    alpha = 0.3
    plt.text(dx[1]+0.8e-3, alpha*linf_err[1]+(1-alpha)*linf_err[2], r"$2$", fontsize=12)
    
    alpha = 0.4
    plt.text(alpha*dx[1]+(1.0-alpha)*dx[2], linf_err[2] - 3.0e-4, r"$1$", fontsize=12)

    # Maximize window
    plt.get_current_fig_manager().window.showMaximized()

    # Save plot to disk
    output_filename = "{:s}/error_grid_spacing.png".format(out_dir)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')

    # Close plot window
    plt.close()

    return None

def get_last_line(file_path):
    
    with open(file_path, 'rb') as file:
        # Go to the end of the file (2nd arg `2` means "seek from end")
        file.seek(-2, 2)  
        
        # Keep moving backward until a newline is found
        while file.read(1) != b'\n':
            file.seek(-2, 1)
        last_line = file.readline().decode('utf-8')
    return last_line

def get_run_time(logfilename):

    last_line = get_last_line(logfilename)
    last_line = last_line.rstrip().split(" ")
    
    compute_time = float(last_line[-1])

    return compute_time

def main():

    domain = [0.0, 1.0, 0.0, 1.0]

    # domain lengths
    Lx = domain[1] - domain[0]
    Ly = domain[3] - domain[2]

    # set boundary values
    bc     = [0.0, 0.0, 1.0e2, 0.0]
    nterms = 51 # number of terms to be used for the analytical solution (do not change)

    # grid sizes
    ngrids_list = sys.argv[1]
    ngrids = list(map(int,ngrids_list.split(",")))
    nruns  = len(ngrids)

    # result directory name
    result_dir = sys.argv[2]

    # Device type - cpu/gpu
    device = sys.argv[3]

    l2_errors    = np.zeros(nruns)
    linf_errors  = np.zeros(nruns)
    dx_grid      = np.zeros(nruns)
    compute_time = np.zeros(nruns)

    for irun in range(nruns):

        nx = ngrids[irun]
        dx_grid[irun] = float(Lx)/float(nx)
        out_dir="{:s}/nx_{:d}".format(result_dir,nx)
        filename = "{:s}/laplace_solution_{:s}_nx_{:d}_ny_{:d}.txt".format(out_dir,device,nx,nx,nx)
        x,y,sol = read_laplace_solution(filename)

        analytical_values = analytical_laplace_solution(x,y,domain,bc,nterms)
        error, max_error, max_ind = compute_error(sol,analytical_values)

        run_logfile  = "{:s}/run_nx_{:d}.log".format(out_dir,nx)
        compute_time[irun] = get_run_time(run_logfile)

        # Reshape data for plot output
        contour_data = np.column_stack([x,y,sol,error,analytical_values])
        
        print("--> Running grid {:2d}/{:d} (dx = {:8.4e}), Max error = {:17.9e}, Max Index = {:2d}, Coords = ({:17.9e},{:17.9e})"
              .format(irun+1,nruns,dx_grid[irun],max_error,max_ind,x[max_ind],y[max_ind]),flush=True)

        # Save contour plot as PNG
        save_contour_plot(contour_data,nx,out_dir,bc[2])

        # Output error to dat file
        output_error_to_file(x,y,error,nx,out_dir)

        l2_errors[irun], linf_errors[irun] = compute_error_norms(error)

    l2_order, linf_order = compute_order_of_accuracy(dx_grid,l2_errors,linf_errors)

    # Plot error with grid-spacing
    plot_error_dx(dx_grid,l2_errors,linf_errors,result_dir)

    avg_l2_order   = np.mean(l2_order)
    avg_linf_order = np.mean(linf_order)

    l2_order   = np.insert(l2_order  , 0, 0, axis=0)
    linf_order = np.insert(linf_order, 0, 0, axis=0)

    data = np.column_stack([ngrids, compute_time, dx_grid, l2_errors, l2_order, linf_errors, linf_order])

    headers        = ["Grid size (Nx)", "Run time (s)", "dx","L2 error norms","L2 order","Linf error norms", "Linf order"]
    data_alignment = ("center",)*len(headers)

    table = tabulate(data, headers=headers,colalign=data_alignment,
                     tablefmt="grid", floatfmt=(".0f","8.5f","6.4f","6.4e","4.2f","6.4e","4.2f"))

    print(table,flush=True)

    # Check order of accuracy to given tolerance
    err_tol          = 5.0  # Error tolerance in %
    analytical_order = 2.0  # Order of accuracy for numerical schemes
    tol_val          = 0.01*err_tol*analytical_order   # Absolute value of tolerance

    # Print info messages
    print("[I] Average order of accuracy: ")
    print("--> L2  : {:5.2f} (Analytical Value = {:5.2f})".format(avg_l2_order,analytical_order))
    print("--> Linf: {:5.2f} (Analytical Value = {:5.2f})".format(avg_linf_order, analytical_order))

    # Set default error code
    return_code = 1

    # If order of accuracy is as expected, set success return code
    if (np.abs(avg_l2_order - analytical_order) < tol_val and np.abs(avg_linf_order - analytical_order) < tol_val):
        return_code = 0
    
    return return_code

sys.exit(main())