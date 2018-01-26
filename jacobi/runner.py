import subprocess
from multiprocessing import cpu_count
import os
import time

# FOR WINDOWS
executable = "mpiexec -n %d python jacobi.py A%s.txt B%s.txt %.0e"
if os.name == "posix":
    executable = "mpirun -n %d python3 jacobi.py A%s.txt B%s.txt %.0e"
cores = (1, 2,)
matrix_sizes = (8,)
accuracies = (1e-3,)


DEVNULL = open(os.devnull, "w")

for core in cores:
    print("%d cores..." % core, end="")
    for size in matrix_sizes:
        print("%d " % size, end="")
        for eps in accuracies:
            exec = executable % (core, size, size, eps)
            process = subprocess.Popen(exec, stdout=DEVNULL, shell=True)
            process.communicate()
    print()

