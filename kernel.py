import numpy as np
import matplotlib.pyplot as plt
import h5py as h
from sklearn.neighbors import KernelDensity
from glob import iglob


def find_peaks(f, x):
    x = x.reshape(len(x))
    df = (f[1:] - f[:-1]) / (x[1] - x[0])
    ddf = df[1:] - df[:-1]
    zeros = np.argsort(np.abs(df))
    peaks = []
    for zero in zeros:
        if zero > 50 and zero < len(x) - 50: # Avoid the flat curves on the sides
            if ddf[zero] < 0:
                peaks.append(zero)

    peaks = np.array(peaks)
    if len(peaks) == 1:
        return peaks
    else:
        mean_peaks = np.mean(peaks)

        upper = peaks[peaks >= mean_peaks]
        lower = peaks[peaks < mean_peaks]

        f_upper = [f[i] for i in upper]
        f_lower = [f[i] for i in lower]

        upper_max = upper[np.argmax(f_upper)]
        lower_max = lower[np.argmax(f_lower)]

        return [lower_max, upper_max]


def bimodal_means(data):

    X_plot = np.linspace(np.min(data), np.max(data), 1000)[:, np.newaxis]

    kde = KernelDensity(kernel="gaussian", bandwidth=0.005).fit(data[:, np.newaxis])
    log_dens = kde.score_samples(X_plot)

    fit = np.exp(log_dens)

    peaks = find_peaks(fit, X_plot)

    mean = np.mean(data)

    lower_peak = X_plot[np.min(peaks)]
    upper_peak = X_plot[np.max(peaks)]

    return [mean, lower_peak[0], upper_peak[0]]


def batch_estimate(data,observable,k):
    '''Devide data into k batches and apply the function observable to each.
    Returns the mean and standard error.'''
    batches = np.reshape(data,(k,-1))
    values = np.apply_along_axis(observable, 1, batches)
    return np.mean(values, axis = 0), np.std(values, axis = 0)/np.sqrt(k-1)


def adjusted_mean(data, batches = 10):
    means, errs = batch_estimate(data, bimodal_means, batches)
    return means[0], np.abs(means[0] - means[1]) + errs[1], np.abs(means[0] - means[2]) + errs[2]


data_sets = []
betas = []
for fname in iglob("C:\\Users\\Casper\\Desktop\\project\\Simulation\\DATA\\Data_fig1\\data_w*_b*.hdf5"): # Hier zitten alle data files, die moet je zelf ff maken want die kan ik niet echt sturen.
    with h.File(fname, 'r') as f:
        data_sets.append(f["action_measurements"][()][0])
        betas.append(f["action_measurements"].attrs["beta"])

means = []
upper_errors = []
lower_errors = []
for i, set in enumerate(data_sets):
    
    mean, lower_error, upper_error = adjusted_mean(set[int(len(set) / 10):], batches = 10)
    means.append(mean)
    lower_errors.append(lower_error)
    upper_errors.append(upper_error)

data = np.array([means, lower_errors, upper_errors])
np.save("data", data)

# Eerst resultaten uitrekenen en dan opslaan want het duurt best even.
exit()
means, lower_errors, upper_errors = np.load('data.npy')

plt.xlim(0.88, 1.13)
plt.ylim(0.2, 0.6)
plt.scatter(betas, means, color = 'black', s = 0.1)
plt.errorbar(betas, means, yerr=[lower_errors, upper_errors], fmt= 'none', color = 'red', capsize = 1, capthick = 1, elinewidth = 0.1)
plt.show()