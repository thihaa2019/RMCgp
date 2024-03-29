from scipy.stats import qmc
import numpy as np

class design_generator():
    def __init__(self,N_samples,X_lower,X_upper,I_lower,I_upper):
        assert N_samples >0, "Number of samples cannot be 0 or less"
        self.N_samples = N_samples
        self.X_lower   = X_lower
        self.X_upper   = X_upper
        self.I_lower   = I_lower
        self.I_upper   = I_upper

    @property
    def create_samples(self):
    # samplings
        eps = 1e-1
        sampler = qmc.LatinHypercube(d=2, scramble=True)
        if np.abs(self.X_lower-self.X_upper)<=eps:
            self.X_upper = self.X_upper+eps
        if self.X_lower <0:
            self.X_lower = 0
        W = sampler.random(n = self.N_samples)
        l_bounds = [self.X_lower,self.I_lower]
        u_bounds = [self.X_upper,self.I_upper]
        W= qmc.scale(W, l_bounds, u_bounds)
        X_designs = W[:,0]
        I_designs = W[:,1]
        return X_designs,I_designs
    
