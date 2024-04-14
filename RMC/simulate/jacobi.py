from .sim import Sim
import numpy as np

class Jacobi(Sim):
    def __init__(self,X0,nstep,nsim,maturity,mean_rev_rate,mean_rev_levels,vol,UB,LB,noises):
        self.mean_rev_rate = mean_rev_rate
        self.mean_rev_levels = mean_rev_levels
        self.sigmas = vol

        super(Jacobi,self).__init__(X0,nstep,nsim,maturity,(self.mean_rev_rate,self.mean_rev_levels),self.drift_func,\
                                (self.sigmas,self.mean_rev_levels),self.diffusion_func,UB,LB,noises)


    @staticmethod
    def drift_func(alpha_t,m_t,x_t):
        return  alpha_t * (m_t- x_t)

    @staticmethod
    def diffusion_func(sigma_t,m_t,x_t):
        return sigma_t *(x_t) * (1-x_t)
