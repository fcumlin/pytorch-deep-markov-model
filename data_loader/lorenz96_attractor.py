import numpy as np
import scipy.integrate
import gin


def dB_to_lin(dB):
    """Convert dB value to linear scale."""
    return 10 ** (dB / 10.0)


def L96(t, x, N=20, F_mu=8, sigma_e2=.1):
    """Lorenz 96 model with constant forcing
    Adapted from: https://www.wikiwand.com/en/Lorenz_96_model 
    """
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    F_N = np.random.normal(loc=F_mu, scale=np.sqrt(sigma_e2), size=(N,))
    for i in range(N):
        #print(F_N[i])
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F_N[i]
    return d


@gin.configurable
class Lorenz96SSMOld:
    """ 
    This class defines the state space model for the high-dimensional Lorenz-96 attractor (Lorenz96SSM). The high-dimensional
    attractor is simulated using Runge-Kutta method (RK45) which uses the dynamics function defined using
    `L96`. The driving noise is incorporated in the forcing function of the attractor for the process / state trajectory 
    instead of conventional additive process noise (as in LinearSSM / LorenzSSM).
    """
    def __init__(
        self,
        n_states=20,
        n_obs=20,
        delta=0.01,
        delta_d=0.005,
        F_mu=8,
        decimate=False,
        mu_w=None,
        H=None,
        method='RK45',
        nonlinear_H=None
    ):
        
        self.n_states = n_states
        self.delta = delta
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.F_mu = F_mu
        self.H = nonlinear_H if nonlinear_H is not None else lambda x: x
        self.mu_w = mu_w if mu_w is not None else np.zeros((n_obs,))
        self.method = method
    
    def h_fn(self, x):
        """ 
        Linear measurement setup y = x + w
        """
        return self.H(x)
    
    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T_time, sigma_e2_dB):

        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        x0 = self.F_mu * np.ones(self.n_states)  # Initial state (equilibrium)
        x0[0] += self.delta  # Add small perturbation to the first variable
        sol = scipy.integrate.solve_ivp(
            L96, 
            t_span=(0.0, T_time), 
            y0=x0, 
            args=(self.n_states, self.F_mu, self.sigma_e2,), 
            method=self.method, 
            t_eval=np.arange(0.0, T_time, self.delta), 
            max_step=self.delta
        )
    
        x_lorenz = np.concatenate((sol.y.T, x0.reshape((1, -1))), axis=0)
        assert x_lorenz.shape[-1] == self.n_states, "Shape mismatch for generated state trajectory"
        
        T = x_lorenz.shape[0]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            x_lorenz_d = x_lorenz[0:T:K,:]
        else:        
            x_lorenz_d = np.copy(x_lorenz)

        return x_lorenz_d[:-1]

    def generate_measurement_sequence(self, T, x_lorenz, smnr_dB=10.0):
        
        #signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        signal_p = np.var(self.h_fn(x_lorenz))
        #print("Signal power: {:.3f}".format(signal_p))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_lorenz = np.zeros((T, self.n_obs))
        
        #print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))
        
        #print(self.H.shape, x_lorenz.shape, y_lorenz.shape)
        for t in range(0,T):
            y_lorenz[t] = self.h_fn(x_lorenz[t]) + w_k_arr[t]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            y_lorenz_d = y_lorenz[0:T:K,:] 
        else:
            y_lorenz_d = np.copy(y_lorenz)

        return y_lorenz_d
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB, num=None):
        del num  # Unused as a new sequence is generated each time.
        
        T_time = T * self.delta
        x_lorenz = self.generate_state_sequence(T_time=T_time, sigma_e2_dB=sigma_e2_dB)
        y_lorenz = self.generate_measurement_sequence(T=T, x_lorenz=x_lorenz, smnr_dB=smnr_dB)

        return x_lorenz, y_lorenz

    def get_next_state_distribution(self, x_prev):
        """
        Get the mean and covariance of the next state given the previous state.
        
        Args:
            x_prev: Previous state vector of shape (n_states,)
            
        Returns:
            tuple: (mean, cov) where
                - mean: Expected next state of shape (n_states,)
                - cov: Covariance matrix of shape (n_states, n_states)
        """
        # For the Lorenz96 system, the mean is computed by integrating the deterministic part
        # and the covariance comes from the stochastic forcing
        if x_prev.ndim == 1:
            x_prev = x_prev[None, :]
        
        # Compute deterministic part (no forcing)
        def L96_deterministic(t, x):
            d = np.zeros(self.n_states)
            for i in range(self.n_states):
                d[i] = (x[(i + 1) % self.n_states] - x[i - 2]) * x[i - 1] - x[i]
            return d
        
        b_size = x_prev.shape[0]
        means = np.zeros_like(x_prev)
        for b in range(b_size):
            cur_x = x_prev[b]
            # Integrate deterministic dynamics for one time step
            sol = scipy.integrate.solve_ivp(
                L96_deterministic,
                t_span=(0.0, self.delta),
                y0=cur_x,
                method=self.method,
                max_step=self.delta  # Use smaller internal steps for accuracy
            )
            
            # Mean is the result of deterministic integration
            mean = sol.y[:, -1]
            means[b] = mean
        
        cov = 0.1 * np.eye(self.n_states)
        
        return mean, cov    


@gin.configurable
class Lorenz96SSM:
    """ 
    This class defines the state space model for the high-dimensional Lorenz-96 attractor (Lorenz96SSM). The high-dimensional
    attractor is simulated using Runge-Kutta method (RK45) which uses the dynamics function defined using
    `L96`. The driving noise is incorporated in the forcing function of the attractor for the process / state trajectory 
    instead of conventional additive process noise (as in LinearSSM / LorenzSSM).
    """
    def __init__(
        self,
        n_states=20,
        n_obs=20,
        delta=0.01,
        delta_d=0.005,
        F_mu=8,
        decimate=False,
        mu_w=None,
        H=None,
        method='RK45',
        nonlinear_H=None
    ):
        
        self.n_states = n_states
        self.delta = delta
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.F_mu = F_mu
        self.H = nonlinear_H if nonlinear_H is not None else lambda x: x
        self.mu_w = mu_w if mu_w is not None else np.zeros((n_obs,))
        self.method = method
    
    def h_fn(self, x):
        """ 
        Linear measurement setup y = x + w
        """
        return self.H(x)
    
    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T_time, sigma_e2_dB, K=1):

        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        x0 = self.F_mu * np.ones(self.n_states)  # Initial state (equilibrium)
        x0[0] += self.delta  # Add small perturbation to the first variable
        sol = scipy.integrate.solve_ivp(
            L96, 
            t_span=(0.0, T_time), 
            y0=x0, 
            args=(self.n_states, self.F_mu, self.sigma_e2,), 
            method=self.method, 
            t_eval=np.arange(0.0, T_time, self.delta), 
            max_step=self.delta
        )
    
        x_lorenz = np.concatenate((sol.y.T, x0.reshape((1, -1))), axis=0)
        assert x_lorenz.shape[-1] == self.n_states, "Shape mismatch for generated state trajectory"
        
        T = x_lorenz.shape[0]
        
        if self.decimate == True:
            x_lorenz_d = x_lorenz[0:T:K,:]
        else:        
            x_lorenz_d = np.copy(x_lorenz)

        return x_lorenz_d[:-1:K]

    def generate_measurement_sequence(self, T, x_lorenz, smnr_dB=10.0):
        
        #signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        signal_p = np.var(self.h_fn(x_lorenz))
        #print("Signal power: {:.3f}".format(signal_p))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_lorenz = np.zeros((T, self.n_obs))
        
        #print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))
        
        #print(self.H.shape, x_lorenz.shape, y_lorenz.shape)
        for t in range(0,T):
            y_lorenz[t] = self.h_fn(x_lorenz[t]) + w_k_arr[t]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            y_lorenz_d = y_lorenz[0:T:K,:] 
        else:
            y_lorenz_d = np.copy(y_lorenz)

        return y_lorenz_d
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB, num=None, K=1):
        del num  # Unused as a new sequence is generated each time.
        
        T_time = T * self.delta * K
        x_lorenz = self.generate_state_sequence(T_time=T_time, sigma_e2_dB=sigma_e2_dB, K=K)
        y_lorenz = self.generate_measurement_sequence(T=T, x_lorenz=x_lorenz, smnr_dB=smnr_dB)

        return x_lorenz, y_lorenz

    def get_next_state_distribution(self, x_prev, K=1):
        """
        Get the mean and covariance of the next state given the previous state.
        
        Args:
            x_prev: Previous state vector of shape (n_states,) or (batch_size, n_states)
            K: Decimation factor - number of time steps to advance (default=1)
                
        Returns:
            tuple: (mean, cov) where
                - mean: Expected next state of shape (n_states,) or (batch_size, n_states)
                - cov: Covariance matrix of shape (n_states, n_states)
        """
        # For the Lorenz96 system, the mean is computed by integrating the deterministic part
        # and the covariance comes from the stochastic forcing
        if x_prev.ndim == 1:
            x_prev = x_prev[None, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute deterministic part (no forcing)
        def L96_deterministic(t, x):
            d = np.zeros(self.n_states)
            for i in range(self.n_states):
                d[i] = (x[(i + 1) % self.n_states] - x[i - 2]) * x[i - 1] - x[i]
            return d
        
        b_size = x_prev.shape[0]
        means = np.zeros_like(x_prev)
        
        # Time span for K steps
        time_span = K * self.delta
        
        for b in range(b_size):
            cur_x = x_prev[b]
            # Integrate deterministic dynamics for K time steps
            sol = scipy.integrate.solve_ivp(
                L96_deterministic,
                t_span=(0.0, time_span),
                y0=cur_x,
                method=self.method,
                max_step=self.delta  # Use smaller internal steps for accuracy
            )
            
            # Mean is the result of deterministic integration
            mean = sol.y[:, -1]
            means[b] = mean
        
        # Covariance scales with the number of time steps (K)
        # This accounts for accumulated uncertainty over K steps
        cov = K * self.sigma_e2 * np.eye(self.n_states)
        
        if squeeze_output:
            return means.squeeze(), cov
        else:
            return means, cov

def main():
    model = Lorenz96SSM(
        n_states=3,
        n_obs=3,
        delta=0.01,
        delta_d=0.005,
        decimate=False,
        method='RK45',
        F_mu=8.0
    )
    states, obs = model.generate_single_sequence(
        T=100,
        sigma_e2_dB=-10.0 - 10*np.log10(5),
        smnr_dB=10.0,
        K=5
    )
    next = model.get_next_state_distribution(states[0], K=5)
    print("Next state mean:", next)
    print("States shape:", states.shape)
    print("Observations shape:", obs.shape)
    print(1/0)
    losses = 0.0
    N = 100
    print(1/0)
    for _ in range(N):
        state, obs = model.generate_single_sequence(
            T=1000,
            sigma_e2_dB=-10.0,
            smnr_dB=10.0
        )
        obs = np.zeros_like(state)
        losses += 10 * np.log10(np.mean((obs - state)**2) / np.mean(state**2))
    print(f"Loss: {losses / N:.2f} dB")


if __name__ == "__main__":
    main()