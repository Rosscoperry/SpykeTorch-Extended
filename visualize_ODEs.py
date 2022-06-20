import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    x = np.arange(0, 2000, 25)
    delta_t = np.array([2, 200, 600, 1000])
    theta_rh = 1200
    threshold = 1600
    f = lambda u, d: -u + d*np.exp((u-theta_rh)/d)  # EIF ode

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    i = 0
    for d_t in delta_t:
        y = f(x, d_t)
        ax = axs[i // 2, i % 2]  # type: plt.Axes
        ax.plot(x, y)
        ax.vlines(threshold, min(y), abs(min(y)), "r")
        ax.set_title("$\Delta_T$ = "+str(d_t))
        ax.hlines(0, 0, max(x), "gray")
        m = min(y)
        M = 200
        ax.set_xlim(0, max(x)-100)
        ax.set_ylim(min(y), abs(min(y)))
        i += 1
    plt.figlegend(["du", "threshold"], )
    plt.tight_layout(pad=2.4)
    plt.savefig("eif_deltaT_plots.png")
    plt.show()

