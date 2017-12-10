from mpi4py import MPI
import util
import numpy as np
from math import log10


ROOT = "../data/"
STAT = "../stat/"


def buffer(buffer_size):
    return np.zeros(int(buffer_size))


def to_buffer(number):
    return np.asarray([number])


def matrix_to_vector(matrix):
    return np.asarray(matrix).ravel()


def save_stat(n_threads, n_matrix, accuracy, execution_time):
    file_name = STAT + "%d %d %d.txt" % (n_threads, n_matrix, log10(accuracy))
    with open(file_name, "a") as file:
        print(execution_time, file=file)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()
n = buffer(1)
eps = buffer(1)
a, b, est = [0] * 3  # needs to be defined

# If main process
if rank == 0:
    start = MPI.Wtime()
    print("Collecting and validating data...", end="")
    try:
        mat_file, est_file, eps = util.read_config()
        a, b, est = util.read_data_and_validate(mat_file, est_file, root=ROOT)

        # converting to one-dimensional array
        a = matrix_to_vector(a)
        # converting to numpy arrays
        b = np.asarray(b)
        est = np.asarray(est)
        eps = to_buffer(eps)
        n[0] = len(b)
    except Exception as e:
        util.print_exception(e)
        MPI.Finalize()
        exit(-1)

    end = MPI.Wtime()
    print("done in %.2f" % ((end - start) * 1000), "ms")
    print("Data exchange...")

# broadcast epsilon and n
comm.Bcast([eps, MPI.DOUBLE], root=0)
comm.Bcast([n, MPI.INT], root=0)

# extract value from array ([1.2] => 1.2)
eps = np.asscalar(eps)
n = int(np.asscalar(n))
chunk = n // pool_size

# buffers for each process's parts
local_a = buffer(n * chunk)
local_b = buffer(chunk)
local_x = buffer(chunk)
if rank != 0:
    est = buffer(n)

# broadcast estimation
comm.Bcast(est, root=0)

# give part of A and B to each process
comm.Scatter(a, local_a, root=0)
comm.Scatter(b, local_b, root=0)

if rank == 0:
    print("Solving with", pool_size, "workers... ", end="")
    start = MPI.Wtime()

# buffer for max norm
max_norm = buffer(1)

while True:
    # count estimation for step
    for i in range(chunk):
        local_x[i] = local_b[i]
        for j in range(n):
            if rank * chunk + i != j:
                local_x[i] -= local_a[i * n + j] * est[j]
        local_x[i] /= local_a[rank * chunk + i + i * n]

    # count max norm for worker
    norm = np.absolute(est[rank * chunk:rank * chunk + chunk] - local_x).max()
    norm = to_buffer(norm)

    # get max norm from all workers
    comm.Reduce(norm, max_norm, op=MPI.MAX, root=0)

    comm.Bcast(max_norm, root=0)

    # collect all estimations into single array
    comm.Allgather(local_x, est)
    if np.asscalar(max_norm) <= eps:
        break

comm.Barrier()
if rank == 0:
    end = MPI.Wtime()
    ms = (end - start) * 1000
    print("done in %.2f" % (ms), "ms")
    save_stat(pool_size, n, eps, ms)


MPI.Finalize()
