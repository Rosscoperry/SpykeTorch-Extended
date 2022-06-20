from SpykeTorch.data_utils import NeuromorphicDataFeed
from SpykeTorch.neurons import EIF, LIF, QIF, AdEx, Izhikevich, LIF_ode
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import torch
import time


def visualize_dataset(path, dt, image_size, backend="TkAgg", loader=None):
    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    if loader is None:
        loader = NeuromorphicDataFeed(path, dt, image_size)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = None
    plt.ion()
    # plt.show()
    for c, data in enumerate(loader):
        spikes, file = data
        # spikes *= 20
        ax.set_title("image " + (file if file is not None else ""))
        plt.yticks([])
        plt.xticks([])
        img = spikes.cpu().numpy().squeeze()
        img *= dt*255
        if im is None:
            im = ax.imshow(img)
            # fig.show()
        else:
            im.set_data(img)

        # im.axes.figure.canvas.draw()
        # im.axes.figure.canvas.flush_events()
        # plt.waitforbuttonpress()
        plt.pause(0.01)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    eif_params = {
        "C": 0.281,
        "delta_t": 2,
        "tau_rc": 0.002,  # 0.00936,
        "theta_rh": -50,
        "threshold": -40,
        "v_reset": None,
        "resting_potential": -65
    }
    adex_adapting_params = {
        "C": 0.00000000004,  # 0.281,
        "delta_t": 0.002,  # V
        "tau_rc": 0.02,  # 0.00936,  # seconds
        "theta_rh": -0.050,  # V
        "threshold": -0.040,  # V
        "v_reset": -0.055,  # V
        "resting_potential": -0.07,  # V
        "a": 0.0,  # nano siemens (nS) -> siemens is ohm^-1
        "b": 5e-12,  # ampere (A) -> e-12 makes it pA
        "tau_w": 0.1,  # seconds
        "refractory_timesteps": 0
    }
    adex_init_burst_params = {
        "C": 0.00000000004,  # 0.281,
        "delta_t": 0.002,  # V
        "tau_rc": 0.01,  # 0.00936,  # seconds | the ref. says 0.005, but seems too low.
        "theta_rh": -0.050,  # V
        "threshold": -0.040,  # V
        "v_reset": -0.051,  # V
        "resting_potential": -0.07,  # V
        "a": 0.5e-9,  # S -> siemens is ohm^-1
        "b": 7e-12,  # ampere (A) -> e-12 makes it pA
        "tau_w": 0.1,  # seconds
        "refractory_timesteps": 0
    }
    adex_burst_params = {
        "C": 0.00000000004,  # 0.281,
        "delta_t": 0.002,  # V
        "tau_rc": 0.01,  # 0.00936,  # seconds | the ref. says 0.005, but seems too low.
        "theta_rh": -0.050,  # V
        "threshold": -0.040,  # V
        "v_reset": -0.046,  # V
        "resting_potential": -0.07,  # V
        "a": -0.5e-9,  # S -> siemens is ohm^-1
        "b": 7e-12,  # ampere (A) -> e-12 makes it pA
        "tau_w": 0.1,  # seconds
        "refractory_timesteps": 0
    }
    adex_delayed_params = {
        "C": 0.00000000004,  # 0.281,
        "delta_t": 0.002,  # V
        "tau_rc": 0.01,  # 0.00936,  # seconds | the ref. says 0.005, but seems too low.
        "theta_rh": -0.050,  # V
        "threshold": -0.040,  # V
        "v_reset": -0.060,  # V
        "resting_potential": -0.07,  # V
        "a": -1.0e-9,  # S -> siemens is ohm^-1
        "b": 10e-12,  # ampere (A) -> e-12 makes it pA
        "tau_w": 0.1,  # seconds
        "refractory_timesteps": 0
    }

    qif_params = {
        "C": 0.281,
        "tau_rc": 0.002,  # 0.00936,
        "u_c": -50,
        "threshold": -40,
        "v_reset": None,
        "resting_potential": -65,
        "a": 0.04
    }
    izhi_params = {
        "threshold": -40.0,
        "resting_potential": -65.0,
        "refractory_timesteps": 0,
        "v_reset": -55.0,
        "d": 4.0,
        "a": 0.02,
        "b": 0.2,
    }
    lif_params = {
        "threshold": -40.0,
        "resting_potential": -65.0,
    }

    lif = LIF(None, **lif_params)
    lif_slow = LIF(None, tau_rc=0.04, **lif_params)
    lif_ode = LIF_ode(None, **lif_params)
    # TODO try to get the same EIF du/dt graph as in neuronal dynamics by tweaking the parameter here
    eif = EIF(None, **eif_params)  # threshold=0.30, theta_rh=200, delta_t=15)
    adex = AdEx(None, **adex_burst_params)  # threshold=0.30, theta_rh=200, delta_t=15)
    qif = QIF(None, **qif_params)
    izhi = Izhikevich(None, **izhi_params)
    spikes_arr = []
    potentials = []
    spikes_arr2 = []
    potentials2 = []
    inputs = []
    dus = []

    def positive_sine(c, dt=0.001):
        if c % 2 == 0:
            t = c*dt
            return torch.ones(1)*(np.sin(2 * np.pi * t/0.024) + 1) / 2
        else:
            return torch.zeros(1)

    filt = lambda t: max(0, np.sin(2 * np.pi * t/0.024 + 7))
    step_current = lambda t: torch.zeros(1) if t < 3 else torch.ones(1)
    rand_spikes = lambda t: torch.zeros(1) if t < 3 else (torch.ones(1) if np.random.rand() >= 0.7 else torch.zeros(1))
    t = 0
    trange = []
    np.random.seed(0)
    snm = lif
    snm2 = lif_slow
    for c, _ in enumerate(range(100)):
        t += 0.001
        trange.append(t)
        inpt = 1550*rand_spikes(_)  # 18*step_current(_)  # torch.zeros(1) if _ < 3 else (5*torch.ones(1) if np.random.rand() >= 0.7 else torch.zeros(1))  # (1.86+0.14+0.5-1.5)*step_current(_)
        inputs.append(inpt)
        inpt = inpt.to(torch.device("cuda"))

        spikes, _, pot, du = snm(inpt, return_thresholded_potentials=True, return_dudt=True,
                                 n_winners=1, return_winners=False)
        # dus.append(du.cpu().numpy()[0])
        spikes_arr.append(spikes.cpu().numpy()[0])
        potentials.append(pot.cpu().numpy()[0])
        spikes, _, pot, du = snm2(inpt, return_thresholded_potentials=True, return_dudt=True,
                                 n_winners=1, return_winners=False)

        spikes_arr2.append(spikes.cpu().numpy()[0])
        potentials2.append(pot.cpu().numpy()[0])

    potentials = np.array(potentials)
    potentials[np.array(spikes_arr, dtype=bool)] = -35
    potentials2 = np.array(potentials2)
    potentials2[np.array(spikes_arr2, dtype=bool)] = -35
    trange = np.asarray(trange)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(3, 1, 1)
    plt.gca().set_title('Input Signal')
    plt.yticks([])
    plt.plot(trange, inputs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Membrane Potential (V)")
    ax = plt.subplot(3, 1, 2)
    ax.set_title('LIF Neuron (Fast Forgetting)')
    ax.vlines(trange[np.flatnonzero(spikes_arr)], -65, -30, colors="r", linestyles="dashed")
    ax.plot(trange, potentials, "b")
    xm, xM = plt.gca().get_xlim()
    plt.hlines(-40, 0, xM, colors="black", linestyles="dashed")
    ax.text(xM, -40, 'V\u209c\u2095', ha='center', va='bottom')
    plt.yticks([])
    # plt.hlines(adex_init_burst_params["theta_rh"], xm, xM, colors="g", linestyles="dotted")
    spikes = np.asarray(spikes_arr) != 0
    plt.xticks(np.asarray(trange)[spikes], labels=None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Membrane Potential (V)")
    # plt.ylim(0, np.max(potentials)+0.1*np.max(potentials))
    ax = plt.subplot(3, 1, 3)
    ax.set_title('LIF Neuron (Slow Forgetting)')
    ax.plot(trange, potentials2, "g")
    ax.vlines(trange[np.flatnonzero(spikes_arr2)], -65, -30, colors="r", linestyles="dashed")
    ax.hlines(-40, 0, xM, colors="black", linestyles="dashed")
    ax.text(xM, -40, 'V\u209c\u2095', ha='center', va='bottom')
    spikes = np.asarray(spikes_arr2) != 0
    plt.gca().set_xticks(np.asarray(trange)[spikes])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Membrane Potential (V)")
    plt.yticks([])
    # plt.gca().set_title('Membrane Potential Variation (du/dt)')
    # plt.plot(trange, dus)
    plt.tight_layout(pad=3.0)
    plt.show()
    plt.waitforbuttonpress()
