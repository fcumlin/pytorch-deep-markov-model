import gin
import numpy as np


def dB_to_lin(x):
    return 10**(x/10)

@gin.configurable
class LinearSSM:
    """ This class defines the Linear state space model (LinearSSM). The design of the transition 
    and measurement matrices were inspired from the KalmanNet paper's linear ssm code:
    https://github.com/KalmanNet/KalmanNet_TSP . Some additional scaling of the transition matrix was
    done for stability reasons using parameters like `beta`. Typically simulated without a driving 
    input.
    """
    def __init__(self, n_states=2, n_obs=2, mu_e=np.zeros(2,), mu_w=np.zeros(2,), gamma=0.8, beta=1.0, drive_noise=False, nonlinear_H=None):
        
        self.n_states = n_states
        self.n_obs = n_obs
        self.gamma = gamma
        self.beta = beta
        if not mu_e is None:
            self.mu_e = mu_e
        else:
            self.mu_e = np.zeros((self.n_states,))
        if not mu_w is None:
            self.mu_w = mu_w
        else:
            self.mu_w = np.zeros((self.n_obs,))
        self.mu_w = mu_w
        self.drive_noise = drive_noise
        self.construct_F()
        self.construct_H()
        self.nonlinear_H = nonlinear_H
        
    def setStateCov(self, sigma_e2):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2):
        self.Cw = sigma_w2 * np.eye(self.n_obs)
    
    def construct_F(self):
        self.F = np.eye(self.n_states) + np.concatenate((np.zeros((self.n_states,1)), 
                                    np.concatenate((np.ones((1,self.n_states-1)), 
                                                    np.zeros((self.n_states-1,self.n_states-1))), 
                                                   axis=0)), 
                                   axis=1)
        
        self.F = self.F * self.gamma
        assert (np.linalg.eig(self.F)[0] <= 1.0).all() == True, "System is not stable!"
        
    def construct_H(self):
        
        self.H = np.rot90(np.eye(self.n_obs)) + np.concatenate((np.concatenate((np.ones((1, self.n_obs-1)), 
                                                              np.zeros((self.n_obs-1, self.n_obs-1))), 
                                                             axis=0), 
                                              np.zeros((self.n_obs,1))), 
                                             axis=1)
        self.H = self.H * self.beta 
        
    def generate_driving_noise(self, k, a=1.0, add_noise=False):
    
        #u_k = np.cos(a*k) # Previous idea (considering start at k=0)
        if add_noise == False:
            u_k = np.cos(a*(k+1)) # Current modification (considering start at k=1)
        elif add_noise == True:
            u_k = np.cos(a*(k+1) + np.random.normal(loc=0, scale=np.pi, size=(1,1))) # Adding noise to the sample
        return u_k

    def generate_state_sequence(self, T, sigma_e2_dB):
        
        self.G = np.ones((self.n_states, 1))
        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        x_arr = np.zeros((T, self.n_states))
        e_k_arr = np.random.multivariate_normal(self.mu_e, self.Ce, size=(T,))
        
        # Generate the sequence iteratively
        for k in range(T-1):

            # Generate driving noise (which is time varying)
            # Driving noise should be carefully selected as per value of k (start from k=0 or =1)
            if self.drive_noise == True: 
                u_k = self.generate_driving_noise(k, a=1.2, add_noise=False)
            else:
                u_k = 0.0

            # For each instant k, sample e_k, w_k
            e_k = e_k_arr[k]

            # Equation for updating the hidden state
            x_arr[k+1] = (self.F @ x_arr[k].reshape((-1,1)) + self.G @ np.array([u_k]).reshape((-1,1)) + e_k.reshape((-1,1))).reshape((-1,))
        
        return x_arr
    
    def generate_measurement_sequence(self, x_arr, T, smnr_dB=10.0):
        if self.nonlinear_H is not None:
            signal_p = np.var(self.nonlinear_H(x_arr))
        else:
            signal_p = np.var(np.einsum('ij,nj->ni', self.H, x_arr)) 
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        y_arr = np.zeros((T, self.n_obs))
        
        #print("sigma_w2: {}".format(self.sigma_w2))
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))

        #print(self.H.shape, x_arr.shape, y_arr.shape)        
        # Generate the sequence iteratively
        for k in range(T):
            # For each instant k, sample e_k, w_k
            w_k = w_k_arr[k]
            # Equation for calculating the output state
            if self.nonlinear_H is None:
                y_arr[k] = self.H @ (x_arr[k]) + w_k
            else:
                y_arr[k] = self.nonlinear_H(x_arr[k]) + w_k
        
        return y_arr
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB, num=None):
        del num  # Unused as a new sequence is generated each time.

        x_arr = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_arr = self.generate_measurement_sequence(x_arr=x_arr, T=T, smnr_dB=smnr_dB)

        return x_arr, y_arr



def main():
    linear_ssm = LinearSSM(2, 2, mu_e=np.zeros(2,), mu_w=np.zeros(2,))
    x, y = linear_ssm.generate_single_sequence(200, -10, 10)
    print(x.shape)
    print(y.shape)
    print(linear_ssm.H)


if __name__ == '__main__':
    main()