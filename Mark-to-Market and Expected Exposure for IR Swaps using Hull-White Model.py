import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np

sigma = 0.1
a = 0.1
timestep = 360
length = 10 # in years
forward_rate = -0.05
day_count = ql.Thirty360()
todays_date = ql.Date(15, 1, 2021)

ql.Settings.instance().evaluationDate = todays_date

spot_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)

hw_process = ql.HullWhiteProcess(spot_curve_handle, a, sigma)
rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)

def generate_paths(num_paths, timestep):
    arr = np.zeros((num_paths, timestep+1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    print(arr[1:2])
    return np.array(time), arr

num_paths = 200
time, paths = generate_paths(num_paths, timestep)
for i in range(num_paths):
    plt.plot(time, paths[i, :], lw=0.8, alpha=0.3)
plt.title("Hull White Simulation")
plt.show()
x = np.empty((num_paths, timestep))
x[:,0] = 0
s = 0
for j in range(num_paths):
    for i in range(timestep):
        x[j, i] = (paths[j, i] - 0.05)*np.power(1/(1.003), i/12)
x_1 = np.empty((num_paths, timestep))
x_1[:, 0] = 0
for i in range(num_paths):
    for j in range(timestep):
        if(x[i,j] >= 0):
            x_1[i, j] = x[i, j]
        else:
            x_1[i, j] = 0
t = np.linspace(0.0, 30, 360)
for i in range(num_paths):
    plt.plot(t, x[i, :], lw=0.8, alpha=0.3)
plt.title("MtM(t)")
plt.show()

for i in range(num_paths):
    plt.plot(t, x_1[i, :], lw=0.8, alpha=0.3)
plt.title("EE(t))")
plt.show()