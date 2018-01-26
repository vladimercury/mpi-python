import subprocess
from multiprocessing import cpu_count
import os
import time
import sys

# FOR WINDOWS
executable = "mpiexec -n %d python main.py %s.txt"
if os.name == "posix":
    executable = "mpirun -n %d python3 main.py %s.txt"
cores = (1, 2, 4,)
matrix_sizes = (1000000, 2500000, 5000000)


OUTPUT = open(os.devnull, "w")
#OUTPUT = sys.stdout

for _ in range(3):
    for core in cores:
        print("\t%d cores..." % core, end="")
        for size in matrix_sizes:
            print("%d " % size, end="")
            exec = executable % (core, size)
            start = time.time()
            process = subprocess.Popen(exec, stdout=OUTPUT, shell=True)
            process.communicate()
            exec_time = time.time() - start
            with open("../qsort-stat/%dx%d.txt" % (core, size), "a") as file:
                print(exec_time, file=file)
        print()

