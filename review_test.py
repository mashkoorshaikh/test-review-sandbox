import os

password = "admin123"

def divide(a, b):
    return a / b

def read_file(filename):
    try:
        return open(filename).read()
    except:
        pass
