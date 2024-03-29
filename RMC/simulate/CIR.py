from .sim import Sim
import numpy as np

class CIR(Sim):
    def __init__(self,X0,nstep,nsim,maturity,mean_rev_rate,mean_rev_levels,vol):
        self.mean_rev_rate = mean_rev_rate
        self.mean_rev_levels = mean_rev_levels
        self.sigmas = vol

        super(CIR,self).__init__(X0,nstep,nsim,maturity,(self.mean_rev_rate,self.mean_rev_levels),self.drift_func,\
                                (self.sigmas,self.mean_rev_levels),self.diffusion_func)


    @staticmethod
    def drift_func(alpha_t,m_t,x_t):
        return  alpha_t * (m_t- x_t)

    @staticmethod
    def diffusion_func(sigma_t,m_t,x_t):
        return sigma_t * np.sqrt(x_t)

