import numpy as np
import matplotlib.pyplot as plt
from Lattice import Lattice
from PhaseVector import PhaseVector
import h5py as h
import argparse
import time

parser = argparse.ArgumentParser(description='Measures the magnetization in the 2D Ising model')
parser.add_argument('-w', type=int, help='Lattice size W')
parser.add_argument('-b', type=float, help='Inverse temperature beta')
parser.add_argument('-n', type=int, help='Number N of measurements (indefinite by default)')
parser.add_argument('-e', type=int, default=10, help='Number E of equilibration sweeps')
parser.add_argument('-m', type=int, default=10, help='Number M of sweeps per measurement')
parser.add_argument('-r', type=bool, default=True, help='Random start configuration (default is True)')
parser.add_argument('-o', type=int, default=30, help='Time in seconds between file outputs')
parser.add_argument('-f', help='Output filename')
args = parser.parse_args()

if args.w is None or args.w < 1:
    parser.error("Please specify a positive lattice size!")
if args.b is None or args.b <= 0.0:
    parser.error("Please specify a positive beta!")


beta = args.b
width = args.w
if args.f is None:
    output_filename = "data_w{}_b{}_{}.hdf5".format(width, beta, time.strftime("%Y%m%d%H%M%S"))
else:
    output_filename = args.f

equilibration_sweeps = args.e
measurement_interval = args.m
time_between_outputs = args.o
random_start = args.r

lattice = Lattice(width, random_start)
for _ in range(equilibration_sweeps):
    lattice.heathbath_update(beta)


measurements = 0
start_time = time.time()
last_output_time = time.time()

with h.File(output_filename, 'a') as f:
    if not "average_actions" in f:
        dataset = f.create_dataset("average_actions", (0,), maxshape = (None,), dtype = np.float32, chunks = True)

        dataset.attrs["parameters"] = str(vars(args))
        dataset.attrs["lattice size"] = width
        dataset.attrs["beta"] = beta
        dataset.attrs["equilibration sweeps"] = args.e
        dataset.attrs["moves per measurement"] = measurement_interval   
        dataset.attrs["start time"] = time.asctime()
    else:
        measurements = len(f["average_actions"])

average_actions = []
while True:
    lattice.heathbath_update(beta)
    average_actions.append(lattice.average_action())
    measurements += 1

    if measurements == args.n or time.time() - last_output_time > time_between_outputs:
        with h.File(output_filename, 'a') as f:
            f["average_actions"].resize(measurements, axis = 0)

            f["average_actions"][-len(average_actions):] = average_actions
            f["average_actions"].attrs["current_time"] = time.asctime()
            average_actions.clear()
        if measurements == args.n:
            break
        else:
            last_output_time = time.time()