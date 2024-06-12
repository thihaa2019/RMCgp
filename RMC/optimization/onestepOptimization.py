from ..costfunctions.finalCosts import final_SOCcontraint,quadratic_SoC_constraint
import scipy
import GPy
import numpy as np

class onestepOptimizer():

    def __init__(self,running_cost,charging_eff,dt):
        
        self.running_cost = running_cost
        self.charging_eff = charging_eff
        self.dt           = dt

    def cost_togo(self,B,X,I,q,*args):
        assert isinstance(B,(list,np.ndarray)), "B must be array of length 1."
        # starting here no B allowed only B[0]
        next_step = np.array([X,I +B[0]*(self.charging_eff*(B[0]>0) + 1/self.charging_eff * (B[0]<0))*self.dt])
        if isinstance(q,final_SOCcontraint) or isinstance(q,quadratic_SoC_constraint):
            q_val = q.cost(next_step[1])
        if isinstance(q,GPy.core.GP):
            q_val = q.predict(next_step.reshape(1,-1))[0].flatten()[0]

        cost = self.running_cost.cost(B[0],X,*args) * self.dt + q_val
        return cost

    def cost_togo_derivative(self,B,X,I,q,*args):

        assert isinstance(B,(list,np.ndarray)), "B must be array of length 1."
        # starting here no B allowed only B[0]
        next_step = np.array([X,I +B[0]*(self.charging_eff*(B[0]>0) + 1/self.charging_eff * (B[0]<0))*self.dt])
        if isinstance(q,final_SOCcontraint) or isinstance(q,quadratic_SoC_constraint):
            q_derivative= q.derivative(next_step[1],B[0])
        if isinstance(q,GPy.core.GP):
            q_derivative = q.predictive_gradients(next_step.reshape(1,-1))[0][:,1].flatten()[0] 
            q_derivative = q_derivative * (self.charging_eff * (B[0]>0) + 1/self.charging_eff * (B[0]<0))

        cost_derivative = self.running_cost.derivative(B[0],X,*args) * self.dt + q_derivative* self.dt
        return cost_derivative

    def optimize(self,X,I,q,*remainingargs):
        target = remainingargs[0]
        starter = X- target
        # excess gen imply charge only >0

        opt_B = scipy.optimize.minimize(self.cost_togo, jac= self.cost_togo_derivative,x0=starter, \
                                       args=(X,I,q,*remainingargs),method = "L-BFGS-B",bounds=None)
        return opt_B.x[0]
