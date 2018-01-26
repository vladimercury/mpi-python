from sys import argv


class Config:
    def __init__(self):
        args = argv[1:]
        self.matrix_file, *extra = args
