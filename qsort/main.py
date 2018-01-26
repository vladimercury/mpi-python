from mpi4py import MPI
from reader import read_data
from util import Config

import numpy as np
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


config = Config()


def count_offset(chunks_src, offset_dest):
    offset_dest[0] = 0
    for i in range(1, size):
        offset_dest[i] = offset_dest[i-1] + chunks_src[i-1]


def count_chunks(data_size, chunks_dest, offset_dest):
    base_chunk_size = int(data_size) // size
    modulo = int(data_size) % size

    chunks_dest.fill(base_chunk_size)
    for i in range(modulo):
        chunks_dest[i] += 1
    count_offset(chunks_dest, offset_dest)


def divide_block(block, less_block, more_block, block_pivot):
    for value in block:
        if value <= block_pivot:
            less_block.append(value)
        else:
            more_block.append(value)


#################################################

if rank == 0:
    print("READING DATA...")

data = None
data_size_buf = np.zeros(1)

if rank == 0:
    data = read_data("../qsort-data/%s" % config.matrix_file)
    #data = np.random.randn(14)
    data_size_buf[0] = len(data)


##################################################

if rank == 0:
    print("DISTRIBUTING DATA...")

chunk_sizes = np.zeros(size)
chunk_offsets = np.zeros(size)

comm.Bcast(data_size_buf, root=0)

count_chunks(data_size_buf[0], chunk_sizes, chunk_offsets)

local = np.zeros(int(chunk_sizes[rank]))
comm.Scatterv([data, chunk_sizes, chunk_offsets, MPI.DOUBLE], local)


#################################################

if rank == 0:
    print("MAIN ALGORITHM...")

iterations = math.ceil(math.log2(size))
pivot = np.zeros(1)
chunk_size_buf = np.zeros(1)

for i in range(iterations - 1, -1, -1):
    if rank == 0:
        print("\t iter %d" % i)
    pair = rank ^ (1 << i)

    if rank < pair:
        if len(local):
            pivot[0] = local[0]
        comm.Send(pivot, dest=pair)
    else:
        comm.Recv(pivot, source=pair)

    own_block = list()
    send_block = list()
    if rank < pair:
        divide_block(local, own_block, send_block, pivot[0])
    else:
        divide_block(local, send_block, own_block, pivot[0])

    own_block = np.asarray(own_block)
    send_block = np.asarray(send_block)

    comm.Isend(send_block, dest=pair, tag=rank)
    status = MPI.Status()
    comm.Probe(source=pair, tag=pair, status=status)
    recv_size = status.count // 8
    recv_block = np.zeros(recv_size)
    comm.Recv(recv_block, source=pair, tag=pair)

    local = np.concatenate((own_block, recv_block))
    comm.Barrier()


###############################################

if rank == 0:
    print("GATHERING DATA...")

if len(local) > 1:
    local.sort()

chunk_size_buf[0] = len(local)
comm.Gather(chunk_size_buf, chunk_sizes, root=0)
if rank == 0:
    count_offset(chunk_sizes, chunk_offsets)
comm.Gatherv(local, [data, chunk_sizes, chunk_offsets, MPI.DOUBLE], root=0)

if rank == 0:
    with open("../res/QSORT-" + config.matrix_file, "w") as file:
        for i in range(len(local)):
            print(local[i], file=file)

