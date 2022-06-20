import torch
import numpy as np
from . import functional as sf
import matplotlib.pyplot as plt
import time
import sys
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

countC = 0
countY = 0


class Neuron(object):

    def __init__(self, ts=0.01, resting_potential=0.0, v_reset=None, threshold=None, refractory_timesteps=0,
                 inhibition_mode="feature"):
        self._threshold = threshold

        self.resting_potential = resting_potential
        self.previous_state = None
        self._per_neuron_thresh = None
        self.refractory_timestes = refractory_timesteps
        self.ts = ts
        self.refractoriness = refractory_timesteps*self.ts
        self.refractory_periods = None
        self.v_reset = v_reset if v_reset is not None else resting_potential
        self.inhibition_mode = inhibition_mode

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def per_neuron_thresh(self):
        return self._per_neuron_thresh

    @per_neuron_thresh.setter
    def per_neuron_thresh(self, value):
        self._per_neuron_thresh = value

    def __str__(self):
        return "Spiking_Neuron_Base_Class"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def get_params(self):
        params = {}
        for k, v in self.__dict__.items():
            if not isinstance(v, torch.Tensor):
                params[k] = v
        return params

    def get_thresholded_potentials(self, current_state):
        """
        General method to get thresholded membrane potentials, i.e. a Tensor where values are != 0.0 if they were above
        the corresponding neuron's threshold.
        Args:
            current_state: membrane potentials of the neurons

        Returns:
            thresholded membrane potentials.
        """
        thresholded = current_state.clone().detach()

        # inhibit where refractoriness is not consumed
        thresholded[self.refractory_periods > 0] = self.resting_potential

        if self.per_neuron_thresh is None:
            self.per_neuron_thresh = torch.ones(current_state.shape, device=DEVICE)*self.threshold
        if self._threshold is None:
            thresholded[:-1] = 0
        else:
            thresholded[thresholded < self.per_neuron_thresh] = 0.0

        return thresholded

    def _f_ode(self, x, I=0):
        raise NotImplementedError()

    def plot_ode(self, figure: plt.Figure = None, ax: plt.Axes = None, current=None):
        """
        Plots the neuron's ODE. If current is given, ODE with and without current are plotted.
        Multiple plots can be stack onto each other by passing the proper figure and axes as an argument.
        Args:
            figure: pyplot.Figure to use for plots. If None, a new one is created.
            ax: pyplot.Axes to use for the plot. If None, a new one is created.
            current: If provided, draws the current ON/OFF plots.

        Returns:
            The figure and axes of the plot.
        """
        f = False
        if figure is None:
            figure = plt.figure()
            f = True
        if ax is None:
            ax = figure.add_subplot()  # type: plt.Axes

        neuron_name = self.__str__().split("_")[0]
        x_points = np.linspace(self.resting_potential, self.threshold + self.threshold / 2, 300)
        y0 = np.array([self._f_ode(x, 0) for x in x_points])
        ax.plot(x_points, y0)
        if current:
            y1 = np.array([self._f_ode(x, current) for x in x_points])
            ax.plot(x_points, y1)
        xmax = ax.get_xlim()[1]
        ax.set_xlim(min(x_points), max(max(x_points), xmax))
        ymin, ymax = ax.get_ylim()
        if f:
            ax.text(self.threshold,  ymin+2, "Vth",  horizontalalignment='left')
            ax.vlines(self.threshold, -5e4, 5e4, "r", linestyles="dashed")
        ax.set_ylim(bottom=min(min(y0), ymin)-5, top=self.threshold)
        ax.hlines(0, min(x_points) - abs(min(x_points) / 4), max(x_points) + abs(max(x_points) / 4), "black",
                  linewidth=.5)
        leg = ax.get_legend()
        labels = [label._text for label in leg.texts] if leg is not None else []
        l = neuron_name
        if current:
            l += " Current OFF"
        labels.append(l)
        if current:
            labels.append(neuron_name+" Current ON")
        ax.legend(labels)
        plt.show()
        return figure, ax

    def finalize_state_update(self, current_state, return_thresholded_potentials=False, return_dudt=False,
                              return_winners=True, n_winners=1):
        """
        Generalized method to save neurons internal state after updates have been calculated, and to calculate
        the return value for the __call__ methods.
        Used to keep the code cleaner.
        Args:
            current_state:
            return_thresholded_potentials: Boolean. Default False.
            return_dudt: Boolean. Default False.
            return_winners: Boolean.
            n_winners: Int. Default 1.

        Returns:
            Tuple. Return values depends on the selected flags.
        """
        # inhibit where refractoriness is not consumed
        current_state[self.refractory_periods > 0] = self.resting_potential
        current_state.clamp_(self.resting_potential, None)

        dudt = current_state - self.previous_state
        thresholded = self.get_thresholded_potentials(current_state)
        spiked = thresholded != 0.0

        # emitted spikes are scaled by dt
        spikes = torch.div(thresholded.sign(), self.ts)

        winners = sf.get_k_winners(thresholded, kwta=n_winners, inhibition_radius=0, spikes=spikes)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            non_inihibited_spikes[0, w[0], :, :] = True
        current_state[spiked] = self.v_reset

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        self.previous_state = current_state

        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (dudt,)
        if return_winners:
            ret += (winners,)
        return ret


class IF(Neuron):

    def __init__(self, threshold, tau_rc=0.02, ts=0.001, resting_potential=0.0, refractory_timesteps=2, C=0.281):

        """
        Creates an Integrate and Fire neuron(s) that receives input potentials (from a preceding convolution)
        and updates its state according to the amount of PSP received (i.e. if it's enough, it fires a spike).
        The neuron(s) state needs to be manually reset when a sequence of related inputs ends (unless the next input is
        to be considered as related to the current one as well).

        Args:
            threshold: threshold above which the neuron(s) fires a spike
            tau_rc: the membrane time constant.
            ts: the time step used for computations, needs to be at least 10 times smaller than tau_rc.
            resting_potential: potential at which the neuron(s) is set to after a spike.
            refractory_timesteps: number of timestep of hyperpolarization after a spike.
            C: Capacitance of the membrane potential. Influences the input potential effect.
        """

        # assert tau_rc / ts >= 10  # needs to hold for Taylor series approximation

        super(IF, self).__init__(resting_potential=resting_potential, threshold=threshold)
        self.ts = ts
        self.tau_rc = tau_rc
        self.ts_over_tau = ts / tau_rc  # for better performance (compute once and for all)
        self.exp_term = np.exp(-self.ts_over_tau)  # for better performance (compute once and for all)
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self.refractory_periods = None
        self.C = C

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None

    def __str__(self):
        return "IF_Neuron_rt"+str(self.refractory_timesteps)+"_C"+str(self.C)

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1):
        r"""Computes the spike-wave tensor from tensor of potentials.
            Args:
                potentials (Tensor): The tensor of input potentials.
                return_thresholded_potentials (boolean): If True, the tensor of thresholded potentials will be returned
                as well as the tensor of spike-wave. Default: False
            Returns:
                Tuple. (spikes, [thresholded_potentials, ], current_state, [dudt, ], [winners, ]
            """
        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (dudt,)
        if return_winners:
            ret += (winners,)
        # potentials = torch.sum(potentials, (2, 3), keepdim=True)
        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)
            self.time_since_spike = torch.full(potentials.size(), 0.0, device=DEVICE)

        previous_state = self.previous_state.clone().detach()

        current_state = previous_state.float().clone().detach()



        # Input pulses.
        # In the hypothesis that dt << tau_rc (at least one order of magnitude), we can use Taylor's expansion
        # to approximate the exponential function. In this way we can more or less simply add the potentials in.
        # input_spikes_impact = potentials * (1 - self.exp_term)

        input_spikes_impact = potentials*self.ts/self.C  # Taylor expansion form (See Neuronal Dynamics Ch.1 ¶ 1.3.2)
        current_state += input_spikes_impact

        current_state.clip(self.resting_potential, None)

        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0
        spikes = torch.div(thresholded.sign(), self.ts)
        winners = sf.get_k_winners(thresholded, spikes=spikes, kwta=n_winners)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            # inhibit all the feature map
            non_inihibited_spikes[0, w[0], :, :] = True

        current_state[spiked] = self.resting_potential
        current_state[self.refractory_periods > 0] = self.resting_potential
        dudt = current_state - self.previous_state
        self.previous_state = current_state

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness
        self.time_since_spike += self.ts
        self.time_since_spike[spiked] = 0.0

        # emitted spikes are scaled by dt
        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (dudt,)
        if return_winners:
            ret += (winners,)
        return ret


class LIF(Neuron):

    def __init__(self, threshold, tau_rc=0.02, ts=0.001, resting_potential=0.0, refractory_timesteps=2, C=0.281,
                 per_neuron_thresh=None):

        """
        Creates a Leaky Integrate and Fire neuron(s) that receives input potentials from a preceding convolution
        and updates its state according to the amount of 'energy' received (i.e. if it's enough, it fires a spike).
        The neuron(s) state needs to be manually reset when a sequence of related inputs ends (unless the next input is
        to be considered as related to the current one as well).

        Args:
            conv: SpykeTorch.snn.Convolution object the neuron(s) is attached to.
            tau_rc: the membrane time constant.
            ts: the time step used for computations, needs to be at least 10 times smaller than tau_rc.
            resting_potential: potential at which the neuron(s) is set to after a spike.
            threshold: threshold above which the neuron(s) fires a spike
        """

        # assert tau_rc / ts >= 10  # needs to hold for Taylor series approximation

        Neuron.__init__(self, resting_potential=resting_potential, threshold=threshold)
        self.ts = ts
        self.tau_rc = tau_rc
        self.ts_over_tau = ts / tau_rc  # for better performance (compute once and for all)
        self.exp_term = np.exp(-self.ts_over_tau)  # for better performance (compute once and for all)
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self.refractory_periods = None
        self.C = C
        self.per_neuron_thresh = per_neuron_thresh

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None

    def __str__(self):
        return "LIF_Neuron_rt"+str(self.refractory_timesteps)+"_C"+str(self.C)

    def _f_ode(self, x, I=0):
        return -(x - self.resting_potential) + (self.tau_rc/self.C)*I

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1, return_winning_spikes=False):
        r"""Computes the spike-wave tensor from tensor of potentials. If :attr:`threshold` is :attr:`None`, all the neurons
            emit one spike (if the potential is greater than zero) in the last time step.
            Args:
                potentials (Tensor): The tensor of input potentials.
                return_thresholded_potentials (boolean): If True, the tensor of thresholded potentials will be returned
                as well as the tensor of spike-wave. Default: False
            Returns:
                Tensor: Spike-wave tensor.
            """
        # potentials = torch.sum(potentials, (2, 3), keepdim=True)
        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)
            self.time_since_spike = torch.full(potentials.size(), 0.0, device=DEVICE)
            self.time_since_injection = torch.full(potentials.size(), 0.0, device=DEVICE)
            self.potential_at_last_injection = torch.full(potentials.size(), self.resting_potential, device=DEVICE)

        previous_state = self.previous_state.clone().detach()

        # ## adapted from Nengo LIF neuron ## #

        # Exponential decay of the membrane potential.
        # To avoid the need of an extra tensor of time-since-last-spike, we can model it as a difference using
        # a constant exponential time step for the decay and eventually clipping the value to resting_pot.
        # pot = resting_pot + torch.mul((previous_state - resting_pot), np.exp(-dt/tau_rc))
        exp_term = torch.clip(torch.exp(-self.time_since_injection/self.tau_rc), max=1)
        current_state = self.resting_potential + (self.potential_at_last_injection - self.resting_potential) * exp_term

        # Input pulses.
        # In the hypothesis that dt << tau_rc (at least one order of magnitude), we can use Taylor's expansion
        # to approximate the exponential function. In this way we can more or less simply add the potentials in.
        # input_spikes_impact = potentials * (1 - self.exp_term)

        input_spikes_impact = potentials*self.ts/self.C  # Taylor expansion form (See Neuronal Dynamics Ch.1 ¶ 1.3.2)
        current_state += input_spikes_impact
        self.time_since_injection += self.ts
        self.time_since_injection[input_spikes_impact > 0] = self.ts
        # inhibit where refractory period is not yet passed.
        current_state[self.refractory_periods > 0] = self.resting_potential
        current_state.clip(self.resting_potential, None)
        self.potential_at_last_injection[input_spikes_impact > 0] = current_state[input_spikes_impact > 0]
        dudt = current_state - self.previous_state
        # resting = torch.full(potentials.size(), resting_pot)
        # delta = torch.add(potentials, -previous_state)
        # delta = torch.add(-previous_state, resting_pot)
        # exp_term = -np.expm1(-dt / leaky_term)
        # delta = torch.mul(delta, exp_term)
        # current_state = torch.add(previous_state, -delta)
        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0
        spikes = torch.div(thresholded.sign(), self.ts)

        winners = sf.get_k_winners(thresholded, spikes=spikes, kwta=n_winners)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            if self.inhibition_mode == "feature":  # inhibit all the feature map
                non_inihibited_spikes[0, w[0], :, :] = True  # This is then used to inhibit all neurons in the same feature-group of neurons as the one who winned
            elif self.inhibition_mode == "location":
                non_inihibited_spikes[0, :, w[1], w[2]] = True
            # non_inihibited_spikes[0] = True
        current_state[spiked] = self.resting_potential
        self.previous_state = current_state

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness
        self.time_since_spike += self.ts
        self.time_since_spike[spiked] = 0.0

        # emitted spikes are scaled by dt
        ret = (spikes, )
        if return_thresholded_potentials:
            ret += (thresholded, )
        ret += (current_state, )
        if return_dudt:
            ret += (dudt, )
        if return_winners:
            ret += (winners, )
        if return_winning_spikes:
            not_winning_spikes = torch.full(spiked.shape, True)
            for w in winners:
                not_winning_spikes[0, w[0], w[1], w[2]] = False
            ws = spikes.clone()
            ws[not_winning_spikes] = 0.0
            ret += (ws, )
        return ret

class LIF_Simple(Neuron):
    def __init__(self, threshold, tau_rc=0.02, ts=0.001, resting_potential=0.0, refractory_timesteps=2,
                 per_neuron_threshold=None, **kwargs):
        Neuron.__init__(self, resting_potential=resting_potential, threshold=threshold)
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = refractory_timesteps*ts
        self.per_neuron_thresh = per_neuron_threshold
        self.tau_rc = tau_rc
        self.ts = ts
        assert tau_rc > 3*ts  # needed for Taylor approx.; actually would be better with times more than ts
        self.decay = 1 - ts/tau_rc  # Taylor approx

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None

    def __str__(self):
        return "LIF_Simple_RT" + str(self.refractory_timesteps) + "_tau" + str(self.tau_rc)

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1, return_winning_spikes=False):

        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)
        previous_state = self.previous_state.clone().detach()

        current_state = previous_state*self.decay
        current_state += potentials
        current_state[self.refractory_periods > 0] = self.resting_potential

        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0
        spikes = thresholded.sign()

        winners = sf.get_k_winners(thresholded, spikes=spikes, kwta=n_winners)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            if self.inhibition_mode == "feature":  # inhibit all the feature map
                non_inihibited_spikes[0, w[0], :, :] = True  # This is then used to inhibit all neurons in the same feature-group of neurons as the one who winned
            elif self.inhibition_mode == "location":
                non_inihibited_spikes[0, :, w[1], w[2]] = True
            # non_inihibited_spikes[0] = True
        current_state[spiked] = self.resting_potential
        self.previous_state = current_state

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        # emitted spikes are NOT scaled by dt
        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (current_state-previous_state,)
        if return_winners:
            ret += (winners,)
        if return_winning_spikes:
            not_winning_spikes = torch.full(spiked.shape, True)
            for w in winners:
                not_winning_spikes[0, w[0], w[1], w[2]] = False
            ws = spikes.clone()
            ws[not_winning_spikes] = 0.0
            ret += (ws,)
        return ret

class LIF_ode(Neuron):

    def __init__(self, threshold, tau_rc=0.02, ts=0.001, resting_potential=0.0, refractory_timesteps=2, C=0.281,
                 per_neuron_thresh=None):

        """
        Creates a Leaky Integrate and Fire neuron(s) that receives input potentials from a preceding convolution
        and updates its state according to the amount of 'energy' received (i.e. if it's enough, it fires a spike).
        The neuron(s) state needs to be manually reset when a sequence of related inputs ends (unless the next input is
        to be considered as related to the current one as well).

        Args:
            conv: SpykeTorch.snn.Convolution object the neuron(s) is attached to.
            tau_rc: the membrane time constant.
            ts: the time step used for computations, needs to be at least 10 times smaller than tau_rc.
            resting_potential: potential at which the neuron(s) is set to after a spike.
            threshold: threshold above which the neuron(s) fires a spike
        """

        # assert tau_rc / ts >= 10  # needs to hold for Taylor series approximation

        super(LIF_ode, self).__init__(resting_potential=resting_potential, threshold=threshold)
        self.ts = ts
        self.tau_rc = tau_rc
        self.ts_over_tau = ts / tau_rc  # for better performance (compute once and for all)
        self.exp_term = np.exp(-self.ts_over_tau)  # for better performance (compute once and for all)
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self.refractory_periods = None
        self.C = C
        self.per_neuron_thresh = per_neuron_thresh

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None

    def __str__(self):
        return "LIF_ode_Neuron_rt"+str(self.refractory_timesteps)+"_C"+str(self.C)

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1):
        r"""Computes the spike-wave tensor from tensor of potentials. If :attr:`threshold` is :attr:`None`, all the neurons
            emit one spike (if the potential is greater than zero) in the last time step.
            Args:
                potentials (Tensor): The tensor of input potentials.
                return_thresholded_potentials (boolean): If True, the tensor of thresholded potentials will be returned
                as well as the tensor of spike-wave. Default: False
            Returns:
                Tensor: Spike-wave tensor.
            """
        # potentials = torch.sum(potentials, (2, 3), keepdim=True)
        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)

        previous_state = self.previous_state.clone().detach()

        du = (self.resting_potential - previous_state)*self.ts_over_tau
        du += potentials * self.ts / self.C

        current_state = previous_state + du

        # inhibit where refractory period is not yet passed.
        current_state[self.refractory_periods > 0] = self.resting_potential
        current_state.clip(self.resting_potential, None)

        dudt = current_state - self.previous_state

        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0
        spikes = torch.div(thresholded.sign(), self.ts)

        winners = sf.get_k_winners(thresholded, spikes=spikes, kwta=n_winners)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            # inhibit the whole feature map for the next iteration. One feature map = one feature.
            # One neuron firing in one feature map = one feature at that position.
            non_inihibited_spikes[0, w[0], :, :] = True
            # non_inihibited_spikes[0] = True
        current_state[spiked] = self.resting_potential

        self.previous_state = current_state

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        # emitted spikes are scaled by dt
        ret = (spikes, )
        if return_thresholded_potentials:
            ret += (thresholded, )
        ret += (current_state, )
        if return_dudt:
            ret += (dudt, )
        if return_winners:
            ret += (winners, )
        return ret


class EIF(Neuron):

    def __init__(self, threshold, tau_rc=0.02, ts=0.001, delta_t=0.5, theta_rh=None, resting_potential=0.0,
                 refractory_timesteps=2, C=0.281, v_reset=None):
        """
        Creates an exponential integrate and fire neuron.
        Args:
            conv: snn.Convolution object the neuron relates to.
            tau_rc: Membrane time constant a.k.a. tau_m or tau in seconds. Default: 0.02.
            ts: time-step value in seconds. Default: 0.001.
            delta_t: Sharpness parameter (upswing on the exponential curve). If ~0, EIF behaves like LIF. Default: 0.5.
            theta_rh: Rheobase threshold. Default: 5.
            resting_potential: Default: 0.0.
            threshold: Default: None.
        """
        Neuron.__init__(self, resting_potential=resting_potential, threshold=threshold)

        # assert abs(theta_rh / (resting_potential + delta_t)) >= 10, \
        #     "Needs to hold as it is assumed in Neuronal Dynamics book, Ch.5 ¶ 5.2"
        self.tau_rc = tau_rc
        self.ts = ts
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self.refractory_periods = None
        self.C = C
        self.delta_t = delta_t
        if theta_rh is None:  # guess from threshold
            theta_rh = -abs(threshold)*0.25 + threshold  # make the rheobase threshold smaller of the threshold by 25%
        self.theta_rh = theta_rh
        if v_reset is None:
            self.v_reset = resting_potential
        else:
            self.v_reset = v_reset

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None

    def __str__(self):
        return "EIF_Neuron_rt"+str(self.refractory_timesteps)+"_C"+str(self.C)

    def _f_ode(self, x, I=0):
        return -(x - self.resting_potential) + self.delta_t*np.exp((x-self.theta_rh)/self.delta_t) + (self.tau_rc / self.C) * I

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1):
        """
        Calculates the update amount for the neuron as specified by the following differential equation:
        $$
            \tau\frac{du}{dt} = -(u - u_{rest}) + \Delta_T \cdot \exp\left({\frac{u - \Theta_{rh}}{\Delta_T}}\right)
            + R\cdot I(t)
        $$
        Args:
            potentials: Post-synaptic potentials. These are intended to be inside a torch.Tensor object and are the
            equivalent of the sum of the incoming spikes, each scaled by the strength of the synapse (convolution
            weights) they came through.
            return_thresholded_potentials: Default: None.

        Returns:

        """
        # potentials = torch.sum(potentials, (2, 3), keepdim=True)

        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)

        previous_state = self.previous_state.clone().detach()

        du = self.resting_potential - previous_state  # -(previous_state - self.resting_potential)
        du = du + self.delta_t * torch.exp((previous_state - self.theta_rh) / self.delta_t)
        du /= self.tau_rc
        du += potentials/self.C
        du *= self.ts
        current_state = previous_state + du

        # inhibit where refractoriness is not consumed
        current_state[self.refractory_periods > 0] = self.resting_potential
        current_state.clamp_(self.resting_potential, None)

        dudt = current_state - self.previous_state
        # current_state.clip(self.resting_potential, None)

        # TODO: Maybe, given that a proper assumption according to Neuronal Dynamics book, would be to have
        # TODO: threshold >> theta_rh + delta_t, if it is None I could set it to be 1 order of magnitude greater?
        # TODO: i.e. threshold = (delta_T + theta_rh) * 10
        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0

        # emitted spikes are scaled by dt
        spikes = torch.div(thresholded.sign(), self.ts)

        winners = sf.get_k_winners(thresholded, kwta=n_winners, inhibition_radius=0, spikes=spikes)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            non_inihibited_spikes[0, w[0], :, :] = True
        current_state[spiked] = self.v_reset

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        self.previous_state = current_state

        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (dudt,)
        if return_winners:
            ret += (winners,)
        return ret


class EIF_Simple(Neuron):
    def __init__(self, threshold, tau_rc=0.02, ts=0.001, delta_t=0.5, theta_rh=None, resting_potential=0.0,
                 refractory_timesteps=2, v_reset=None, per_neuron_threshold=None, **kwargs):
        Neuron.__init__(self, resting_potential=resting_potential, threshold=threshold)
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self._per_neuron_thresh = per_neuron_threshold
        self.tau_rc = tau_rc
        self.ts = ts
        assert tau_rc > 3*ts  # needed for Taylor approx.; actually would be better with 6 times more than ts
        self.decay = 1 - ts/tau_rc  # Taylor approx

        self.delta_t = delta_t
        if theta_rh is None:  # guess from threshold
            theta_rh = -abs(threshold)*0.25 + threshold  # make the rheobase threshold smaller of the threshold by 25%
        self.theta_rh = theta_rh
        if v_reset is None:
            self.v_reset = resting_potential
        else:
            self.v_reset = v_reset

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None

    @Neuron.per_neuron_thresh.setter
    def per_neuron_thresh(self, value):
        self._per_neuron_thresh = value
        self.theta_rh = 0.75*value

    def __str__(self):
        return "EIF_Simple_RT" + str(self.refractory_timesteps) + "_tau" + str(self.tau_rc)

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1, return_winning_spikes=False):

        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)
        previous_state = self.previous_state.clone().detach()

        current_state = previous_state*self.decay + self.delta_t * torch.exp((previous_state - self.theta_rh) / self.delta_t)
        current_state += potentials
        current_state[self.refractory_periods > 0] = self.resting_potential

        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0
        spikes = thresholded.sign()

        winners = sf.get_k_winners(thresholded, spikes=spikes, kwta=n_winners)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            if self.inhibition_mode == "feature":  # inhibit all the feature map
                non_inihibited_spikes[0, w[0], :, :] = True  # This is then used to inhibit all neurons in the same feature-group of neurons as the one who winned
            elif self.inhibition_mode == "location":
                non_inihibited_spikes[0, :, w[1], w[2]] = True
            # non_inihibited_spikes[0] = True
        # neurons that fired a spike a reset to v_reset regardless of being winners
        membrane_potential = current_state.clone().detach()
        current_state[spiked] = self.v_reset
        self.previous_state = current_state

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        # but only the winners (and the inhibited neurons) are given a refractory period.
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        # emitted spikes are NOT scaled by dt
        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (membrane_potential,)  # (current_state,)
        if return_dudt:
            ret += (current_state-previous_state,)  # this will result in very negative dudt sometimes, can be done better
        if return_winners:
            ret += (winners,)
        if return_winning_spikes:
            not_winning_spikes = torch.full(spiked.shape, True)
            for w in winners:
                not_winning_spikes[0, w[0], w[1], w[2]] = False
            ws = spikes.clone()
            ws[not_winning_spikes] = 0.0
            ret += (ws,)
        return ret

class AdEx(Neuron):

    def __init__(self, threshold, tau_rc=0.02, ts=0.001, delta_t=0.5,
                 a=0.6, b=0.7, tau_w=1,
                 theta_rh=None, resting_potential=0.0,
                 refractory_timesteps=2, C=0.281, v_reset=None):
        """
        Creates an exponential integrate and fire neuron.
        Args:
            conv: snn.Convolution object the neuron relates to.
            tau_rc: Membrane time constant a.k.a. tau_m or tau in seconds. Default: 0.02.
            ts: time-step value in seconds. Default: 0.001.
            delta_t: Sharpness parameter (upswing on the exponential curve). If ~0, EIF behaves like LIF. Default: 0.5.
            theta_rh: Rheobase threshold. Default: 5.
            resting_potential: Default: 0.0.
            threshold: Default: None.
        """
        super(AdEx, self).__init__(resting_potential=resting_potential, threshold=threshold)

        # assert abs(theta_rh / (resting_potential + delta_t)) >= 10, \
        #     "Needs to hold as it is assumed in Neuronal Dynamics book, Ch.5 ¶ 5.2"
        self.tau_rc = tau_rc
        self.ts = ts
        self.delta_t = delta_t
        if theta_rh is None:  # guess from threshold
            theta_rh = -abs(threshold)*0.25 + threshold  # make the rheobase threshold smaller of the threshold by 25%
        self.theta_rh = theta_rh
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self.refractory_periods = None
        self.C = C
        self.R = tau_rc/C
        self.a = a
        self.b = b
        self.tau_w = tau_w
        if v_reset is None:
            self.v_reset = resting_potential
        else:
            self.v_reset = v_reset

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None
        self.w = None

    def __str__(self):
        return "AdEx_Neuron_rt"+str(self.refractory_timesteps)+"_C"+str(self.C)

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1):
        """
        Calculates the update amount for the neuron as specified by the following differential equation:
        TODO: update to match AdEx Eq.
        $$
            \tau\frac{du}{dt} = -(u - u_{rest}) + \Delta_T \cdot \exp\left({\frac{u - \Theta_{rh}}{\Delta_T}}\right)
            + R\cdot I(t)
        $$
        Args:
            potentials: Post-synaptic potentials. These are intended to be inside a torch.Tensor object and are the
            equivalent of the sum of the incoming spikes, each scaled by the strength of the synapse (convolution
            weights) they came through.
            return_thresholded_potentials: Default: None.

        Returns:

        """
        # potentials = torch.sum(potentials, (2, 3), keepdim=True)

        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)
            self.w = torch.full(potentials.size(), 0.0, device=DEVICE)

        previous_state = self.previous_state.clone().detach()

        du = self.resting_potential - previous_state  # -(previous_state - self.resting_potential)
        du = du + self.delta_t * torch.exp((previous_state - self.theta_rh) / self.delta_t)
        currents_impact = (potentials - self.w)*self.R
        du += currents_impact
        du /= self.tau_rc
        du *= self.ts
        current_state = previous_state + du

        # inhibit where refractoriness is not consumed
        current_state[self.refractory_periods > 0] = self.resting_potential
        current_state.clamp_(self.resting_potential, None)

        dudt = current_state - self.previous_state
        # current_state.clip(self.resting_potential, None)

        # TODO: Maybe, given that a proper assumption according to Neuronal Dynamics book, would be to have
        # TODO: threshold >> theta_rh + delta_t, if it is None I could set it to be 1 order of magnitude greater?
        # TODO: i.e. threshold = (delta_T + theta_rh) * 10
        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0

        # Implement the common adaptation variable update for all the neurons
        self.w += (self.a*(current_state-self.resting_potential) - self.w)/self.tau_w*self.ts
        # Add a current jump only where there has been a spike
        self.w[spiked] += self.b  # see https://journals.physiology.org/doi/pdf/10.1152/jn.00686.2005 for
        #                         # why this is simply added in.

        # emitted spikes are scaled by dt
        spikes = torch.div(torch.abs(thresholded.sign()), self.ts)

        winners = sf.get_k_winners(thresholded, kwta=n_winners, inhibition_radius=0, spikes=spikes)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            # non_inihibited_spikes[0, w[0], :, :] = True
            non_inihibited_spikes[0] = True
        current_state[spiked] = self.v_reset

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        self.previous_state = current_state

        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (dudt,)
        if return_winners:
            ret += (winners,)
        return ret


class QIF(Neuron):

    def __init__(self, threshold, tau_rc=0.02, ts=0.001, u_c=None, a=0.001, resting_potential=0.0,
                 refractory_timesteps=2, C=0.281, v_reset=None):
        """
        Creates an exponential integrate and fire neuron.
        Args:
            conv: snn.Convolution object the neuron relates to.
            tau_rc: Membrane time constant a.k.a. tau_m or tau in seconds. Default: 0.02.
            ts: time-step value in seconds. Default: 0.001.
            a: Sharpness parameter (upswing on the parabolic curve). Default: None.
            u_c: 'Rheobase' (similar to theta_rh in EIF in behaviour) threshold. Default: 5.
            resting_potential: Default: 0.0.
            threshold: Default: None.
        """
        Neuron.__init__(self, resting_potential=resting_potential, threshold=threshold)

        # assert abs(theta_rh / (resting_potential + delta_t)) >= 10, \
        #     "Needs to hold as it is assumed in Neuronal Dynamics book, Ch.5 ¶ 5.2"
        self.tau_rc = tau_rc
        self.ts = ts
        if u_c is None:  # guess from threshold
            u_c = -abs(threshold)*0.25 + threshold  # make the 'rheobase' threshold smaller of the threshold by 25%
        self.u_c = u_c
        self.a = a
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self.refractory_periods = None
        self.C = C
        if v_reset is None:
            self.v_reset = resting_potential
        else:
            self.v_reset = v_reset

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None

    def __str__(self):
        return "QIF_Neuron_rt"+str(self.refractory_timesteps)+"_C"+str(self.C)

    def _f_ode(self, x, I=0):
        return self.a*(x - self.resting_potential)*(x-self.u_c) + (self.tau_rc/self.C)*I

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1):
        """
        Calculates the update amount for the neuron as specified by the following differential equation:
        $$
            \tau\frac{du}{dt} = -a_0(u - u_{rest})(u - u_{c}) + R\cdot I(t)
        $$
        Args:
            potentials: Post-synaptic potentials. These are intended to be inside a torch.Tensor object and are the
            equivalent of the sum of the incoming spikes, each scaled by the strength of the synapse (convolution
            weights) they came through.
            return_thresholded_potentials: Default: None.

        Returns:

        """

        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)

        previous_state = self.previous_state.clone().detach()

        # TODO should consider multiplying input by a resistance?
        # du = -(previous_state - self.resting_potential) \
        #      + self.delta_t * torch.exp((previous_state - self.theta_rh) / self.delta_t) \
        #      + potentials

        du = previous_state - self.resting_potential
        du = du * (previous_state - self.u_c) * self.a
        du /= self.tau_rc
        du += potentials/self.C
        du *= self.ts
        current_state = previous_state + du

        # inhibit where refractoriness is not consumed
        current_state[self.refractory_periods > 0] = self.resting_potential
        torch.clip_(current_state, min=self.resting_potential)
        dudt = current_state - self.previous_state

        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0
        spikes = torch.div(thresholded.sign(), self.ts)
        # winners = sf.get_k_winners_davide(thresholded, spikes, self.per_neuron_thresh)
        winners = sf.get_k_winners(thresholded, kwta=n_winners, inhibition_radius=0, spikes=spikes)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            non_inihibited_spikes[0, w[0], :, :] = True
        current_state[spiked] = self.v_reset

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        self.previous_state = current_state

        # emitted spikes are scaled by dt
        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (dudt,)
        if return_winners:
            ret += (winners,)
        return ret


class Izhikevich(Neuron):

    def __init__(self, threshold, tau_rc=0.02, ts=0.001,
                 a=0.02, b=0.2, c=0.0, d=8.0,
                 resting_potential=0.0,
                 refractory_timesteps=2, C=0.281, v_reset=None):
        """
        Creates an exponential integrate and fire neuron.
        Args:
            conv: snn.Convolution object the neuron relates to.
            tau_rc: Membrane time constant a.k.a. tau_m or tau in seconds. Default: 0.02.
            ts: time-step value in seconds. Default: 0.001.
            delta_t: Sharpness parameter (upswing on the exponential curve). If ~0, EIF behaves like LIF. Default: 0.5.
            theta_rh: Rheobase threshold. Default: 5.
            resting_potential: Default: 0.0.
            threshold: Default: None.
        """
        super(Izhikevich, self).__init__(resting_potential=resting_potential, threshold=threshold)

        # assert abs(theta_rh / (resting_potential + delta_t)) >= 10, \
        #     "Needs to hold as it is assumed in Neuronal Dynamics book, Ch.5 ¶ 5.2"
        self.tau_rc = tau_rc
        self.ts = ts
        # if theta_rh is None:  # guess from threshold
        #     theta_rh = -abs(threshold)*0.25 + threshold  # make the rheobase threshold smaller of the threshold by 25%
        # self.theta_rh = theta_rh
        self.refractory_timesteps = refractory_timesteps
        self.refractoriness = self.refractory_timesteps * ts
        self.refractory_periods = None
        self.C = C
        self.R = tau_rc/C
        self.a = a
        self.b = b
        self.d = d
        if v_reset is None:
            self.v_reset = resting_potential
        else:
            self.v_reset = v_reset
        self.c = c if c is not None else self.v_reset

    def reset(self):
        self.previous_state = None
        self.refractory_periods = None
        self.w = None

    def __str__(self):
        return "Izhikevich_Neuron_rt"+str(self.refractory_timesteps)+"_C"+str(self.C)

    def __call__(self, potentials, return_thresholded_potentials=False, return_dudt=False,
                 return_winners=True, n_winners=1):
        """
        Calculates the update amount for the neuron as specified by the following differential equation:
        TODO: update to match AdEx Eq.
        $$
            \tau\frac{du}{dt} = -(u - u_{rest}) + \Delta_T \cdot \exp\left({\frac{u - \Theta_{rh}}{\Delta_T}}\right)
            + R\cdot I(t)
        $$
        Args:
            potentials: Post-synaptic potentials. These are intended to be inside a torch.Tensor object and are the
            equivalent of the sum of the incoming spikes, each scaled by the strength of the synapse (convolution
            weights) they came through.
            return_thresholded_potentials: Default: None.

        Returns:

        """
        # potentials = torch.sum(potentials, (2, 3), keepdim=True)

        if self.previous_state is None:
            self.previous_state = torch.full(potentials.size(), self.resting_potential, device=DEVICE)
            self.refractory_periods = torch.full(potentials.size(), 0.0, device=DEVICE)
            self.w = torch.full(potentials.size(), self.resting_potential*self.b, device=DEVICE)

        previous_state = self.previous_state.clone().detach()

        du = 0.04*previous_state**2 + 5*previous_state + 140
        du += potentials
        du -= self.w
        current_state = previous_state + du

        # inhibit where refractoriness is not consumed
        current_state[self.refractory_periods > 0] = self.resting_potential
        # current_state.clamp_(self.resting_potential, None)

        dw = self.a*(self.b*current_state - self.w)
        dudt = current_state - self.previous_state
        # current_state.clip(self.resting_potential, None)

        thresholded = self.get_thresholded_potentials(current_state)

        spiked = thresholded != 0.0
        # Add a current jump only where there has been a spike
        self.w = self.w + dw
        self.w[spiked] += self.d

        # emitted spikes are scaled by dt
        spikes = torch.div(torch.abs(thresholded.sign()), self.ts)

        winners = sf.get_k_winners(thresholded, kwta=n_winners, inhibition_radius=0, spikes=spikes)
        non_inihibited_spikes = torch.full(spiked.shape, False)
        for w in winners:
            #non_inihibited_spikes[0, w[0], :, :] = True
            non_inihibited_spikes[0] = True
        current_state[spiked] = self.v_reset

        # update refractory periods
        self.refractory_periods[self.refractory_periods > 0] -= self.ts
        self.refractory_periods[non_inihibited_spikes] = self.refractoriness

        self.previous_state = current_state

        ret = (spikes,)
        if return_thresholded_potentials:
            ret += (thresholded,)
        ret += (current_state,)
        if return_dudt:
            ret += (dudt,)
        if return_winners:
            ret += (winners,)
        return ret


class HetherogeneousNeuron(Neuron):

    def __init__(self, conv):
        super().__init__(conv)

    def get_uniform_distribution(self, range, size):
        ones = np.ones(size)
        uniform = np.random.uniform(*range, size=(size[0], size[1], 1, 1))
        uniform = torch.from_numpy(uniform*ones)
        return uniform.to(DEVICE)


class UniformLIF(LIF, HetherogeneousNeuron):

    def __init__(self, threshold, tau_range, ts=0.001, resting_potential=0.0, refractory_timesteps=2, C=0.281,
                 per_neuron_thresh=None):
        LIF.__init__(self, threshold, C=C, refractory_timesteps=refractory_timesteps, ts=ts,
                     resting_potential=resting_potential, per_neuron_thresh=per_neuron_thresh)
        self.threshold = threshold
        self.tau_range = tau_range
        self.taus = None

    def __call__(self, potentials, *args, **kwargs):

        if self.taus is None:
            self.taus = self.get_uniform_distribution(self.tau_range, potentials.shape)
            self.ts_over_tau = self.ts / self.taus.cpu().numpy()  # for better performance (compute once and for all)
            self.exp_term = np.exp(-self.ts_over_tau)  # for better performance (compute once and for all)

        return super(UniformLIF, self).__call__(potentials, *args, **kwargs)


class UniformEIF(EIF, HetherogeneousNeuron):

    def __init__(self, threshold, tau_range, ts=0.001, delta_t=0.5, theta_rh=None, resting_potential=0.0,
                 refractory_timesteps=2, C=0.281, v_reset=None):
        EIF.__init__(self, threshold=threshold, tau_rc=0.02, ts=ts, delta_t=delta_t, theta_rh=theta_rh,
                     resting_potential=resting_potential, refractory_timesteps=refractory_timesteps, C=C,
                     v_reset=v_reset)
        HetherogeneousNeuron.__init__(self, conv)
        self.tau_range = tau_range
        self.taus = None

    def __call__(self, potentials, *args, **kwargs):

        if self.taus is None:
            self.taus = self.get_uniform_distribution(self.tau_range, potentials.shape)

        return super(UniformEIF, self).__call__(potentials, *args, **kwargs)


class UniformQIF(QIF, HetherogeneousNeuron):

    def __init__(self, threshold, tau_range, ts=0.001, u_c=None, a=0.001, resting_potential=0.0,
                 refractory_timesteps=2, C=0.281, v_reset=None):
        QIF.__init__(self, threshold, tau_rc=0.02, ts=ts, u_c=u_c, a=a, resting_potential=resting_potential,
                 refractory_timesteps=refractory_timesteps, C=C, v_reset=v_reset)
        HetherogeneousNeuron.__init__(self, conv)
        self.tau_range = tau_range
        self.taus = None

    def __call__(self, potentials, *args, **kwargs):
        if self.taus is None:
            self.taus = self.get_uniform_distribution(self.tau_range, potentials.shape)

        return super(UniformQIF, self).__call__(potentials, *args, **kwargs)