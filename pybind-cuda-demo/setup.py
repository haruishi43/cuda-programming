import os
import sys
import errno
import numpy
import subprocess


def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise

build = False
clean = False

if 2 <= len(sys.argv):
    if sys.argv[1] == "build" or sys.argv[1] == "build_ext":
        build = True
    elif sys.argv[1] == "clean":
        clean = True
    else:
        print("Undefined command: {}".format(sys.argv[1]))
        exit(1)
else:
    print("Prease specify a command: build, clean")
    exit(1)

if build:
    print("Configuring...")
    mkdir_p("build")
    subprocess.Popen(["cmake",
                      "-DPYTHON_EXECUTABLE=~/.pyenv/versions/anaconda3-5.2.0/bin/python",
                      "-DCMAKE_BUILD_TYPE=Release",
                      ".."
                      ],
                     cwd="build").wait()
    print("Compiling extension...")
    subprocess.Popen(["make", "-j8"], cwd="build").wait()

if clean:
    print("Removing...")
    subprocess.Popen(["rm", "-r", "build", "__pycache__"], cwd=os.path.join("./", os.path.dirname(sys.argv[0]))).wait()