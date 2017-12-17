from mpi4py import MPI
import util
import numpy as np
from math import log10


ROOT = "../data/"
STAT = "../stat/"
RES = "../res/"


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

comm.Bcast(est, root=0)

comm.Scatter(a, local_a, root=0)
comm.Scatter(b, local_b, root=0)

if rank == 0:
    print("Solving with", pool_size, "workers... ", end="")
    start = MPI.Wtime()

max_norm = to_buffer(eps + 1)

while np.asscalar(max_norm) >= eps / n:
    for i in range(chunk):
        local_x[i] = local_b[i]
        for j in range(n):
            if rank * chunk + i != j:
                local_x[i] -= local_a[i * n + j] * est[j]
        local_x[i] /= local_a[rank * chunk + i + i * n]

    norm = np.absolute(est[rank * chunk:rank * chunk + chunk] - local_x).max()
    norm = to_buffer(norm)

    comm.Reduce(norm, max_norm, op=MPI.MAX, root=0)

    comm.Bcast(max_norm, root=0)

    comm.Allgather(local_x, est)

    # if rank == 0:
    #     check = np.zeros(n)
    #     for i in range(n):
    #         check[i] = a[i*n:i*n+n].dot(est)
    #     diff = np.absolute(check - b)
    #     max_norm = diff.max()
    #
    # comm.Bcast(max_norm, root=0)

comm.Barrier()
if rank == 0:
    end = MPI.Wtime()
    ms = (end - start) * 1000
    print("done in %.2f" % ms, "ms")
    save_stat(pool_size, n, eps, ms)

    # writing result to file
    result_file_name = RES + "RES %d %d %d.txt" % (pool_size, n, log10(eps))
    with open(result_file_name, "w") as result_file:
        print("X =", est, file=result_file)

        # correction report
        check = np.zeros(n)
        for i in range(n):
            check[i] = a[i*n:i*n+n].dot(est)
        print("AR =", check, file=result_file)  # actual result
        print("ER = ", b, file=result_file)  # expected result

        diff = np.absolute(check - b)
        print("DIFF =", diff, file=result_file)  # differences
        print("MAX DIFF =", max(diff), file=result_file) # max difference


MPI.Finalize()
