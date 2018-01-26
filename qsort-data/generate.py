from random import random

RANGE = 100

sizes = [
    1000000,
    2500000,
    5000000
]

for size in sizes:
    print("%d" % size)
    with open("%d.txt" % size, "w") as file:
        print("%d" % size, file=file)
        step = size // 10000 if size > 10000 else 1
        for i in range(size):
            print("%f" % random(), file=file)
            if i % step == 0:
                print("\r%.0f%%" % (i / size * 100), end="")
