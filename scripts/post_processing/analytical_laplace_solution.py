import matplotlib.pyplot as pyt
import os, sys
import numpy as np
from tabulate import tabulate

def analytical_solution(x,y,Lx,Ly,bottom_bndy_val,nterms=100):

    pi = np.pi
    
    sol = 0.0

    for iterm in range(1,nterms):
        # compute constant
        sinh_arg = iterm*pi*Lx/Ly
        cos_arg  = pi*iterm
        an   = (1.0 - np.cos(cos_arg))/(np.sinh(sinh_arg))

        # compute (x,y) variation
        sinh_arg2 = iterm*pi*(Ly - y)/Lx
        sin_arg   = iterm*pi*x/Lx

        sol += an*np.sin(sin_arg)*np.sinh(sinh_arg2)

    analytical_value = 2*bottom_bndy_val*sol
    
    #print("nterms = {:d}, analytical val = {:17.9e}".format(nterms,analytical_value))

    # sys.exit()

    return analytical_value

def analytical_laplace_solution(x,y,domain,Lx,Ly,bc,nterms):
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
       print("[E] Sizes of x and y coordiante arrays are not the same. Code exitting!")
       sys.exit(12)

    # loop over all cells and compute analytical solution
    for icell in range(ncells):
        xloc = x[icell]
        yloc = y[icell]

        laplace_solution[icell] = analytical_solution(xloc,yloc,Lx,Ly,bc[2],nterms)

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

def main():

    #os.system("clear")
    domain = [0.0, 1.0, 0.0, 1.0]

    # domain lengths
    Lx = domain[1] - domain[0]
    Ly = domain[3] - domain[2]

    # set boundary values
    bc     = [0.0, 0.0, 1.0, 0.0]
    nterms = 101 #int(sys.argv[1])

    # grid sizes
    ngrids_list = sys.argv[1]
    ngrids = list(map(int,ngrids_list.split(",")))
    nruns  = len(ngrids)

    # result directory name
    result_dir = sys.argv[2]

    l2_errors   = np.zeros(nruns)
    linf_errors = np.zeros(nruns)
    dx_grid     = np.zeros(nruns)

    for irun in range(nruns):

        nx = ngrids[irun]
        dx_grid[irun] = float(Lx)/float(nx - 1)
        out_dir="{:s}/nx_{:d}".format(result_dir,nx)
        filename = "{:s}/laplace_solution_cpu_nx_{:d}_ny_{:d}.txt".format(out_dir,nx,nx,nx)
        x,y,sol = read_laplace_solution(filename)

        analytical_values = analytical_laplace_solution(x,y,domain,Lx,Ly,bc,nterms)
        error, max_error, max_ind = compute_error(sol,analytical_values)

        print("--> Running grid {:d}/{:d} (dx = {:8.4e}), Max error = {:17.9e}, Max Index = {:d}, Coords = ({:17.9e},{:17.9e})"
              .format(irun+1,nruns,dx_grid[irun],max_error,max_ind,x[max_ind],y[max_ind]))

        out_data = np.column_stack((x,y,error))
        outfilename = "{:s}/error_nx_{:d}.dat".format(out_dir,nx)
        np.savetxt(outfilename,out_data)


        l2_errors[irun], linf_errors[irun] = compute_error_norms(error)

    l2_order, linf_order = compute_order_of_accuracy(dx_grid,l2_errors,linf_errors)

    l2_order   = np.insert(l2_order  , 0, 0, axis=0)
    linf_order = np.insert(linf_order, 0, 0, axis=0)

    data = np.column_stack([ngrids, dx_grid, l2_errors, l2_order, linf_errors, linf_order])

    table = tabulate(data, headers=["Grid size","dx","L2 error norms","L2 order","Linf error norms", "Linf order"],tablefmt="grid")

    print(table)

    return 0

sys.exit(main())