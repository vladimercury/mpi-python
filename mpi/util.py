from sys import argv, stderr
import traceback


def print_exception(exc, file=stderr):
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=file)


def read_config():
    args = argv[1:]
    try:
        matrix, estimation, epsilon, *extra = args
        epsilon = float(epsilon)
    except ValueError as err:
        raise RuntimeError(str(err))
    return matrix, estimation, epsilon


def read_matrix(file_name, root, data_type=float):
    matrix = list()
    vector = list()
    with open(root + file_name, "r") as file:
        m, n = map(int, file.readline().split())
        for _ in range(m):
            line = list(map(data_type, file.readline().split()))
            matrix.append(line[:-1])
            vector.append(line[-1])
    return matrix, vector


def read_vector(file_name, root, data_type=float):
    vector = list()
    with open(root + file_name, "r") as file:
        m = int(file.readline())
        for _ in range(m):
            vector.append(data_type(file.readline()))
    return vector


def read_data_and_validate(matrix_file, estimation_file, root=""):
    matrix, vector = read_matrix(matrix_file, root)
    estimation = read_vector(estimation_file, root)
    matrix_size = len(matrix)
    for row in matrix:
        if len(row) != matrix_size:
            raise RuntimeError("Matrix is not square")
        for cell in row:
            if cell == 0:
                raise RuntimeError("Matrix has a 0 value")
    if matrix_size != len(vector) != len(estimation):
        raise RuntimeError("Vectors' size doesn't match matrix size")
    return matrix, vector, estimation
