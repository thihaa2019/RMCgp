from .hybridrunner import runRMC
from ..simulate import Sim
from ..costfunctions import L1,L2,final_SOCcontraint
import GPy
import os
import pickle
class HybridControl(runRMC):
    def __init__(self,nsim,underlying_process,running_cost,final_cost,BESSparameters,batch_size,value_kernel,normalize_value,policy_kernel,normalize_policy):
        
        assert isinstance(underlying_process,Sim), "Underlying must come from RMC.sim"
        assert isinstance(running_cost,(L1,L2)),"Running cost must come from RMC.costfunctions"
        assert isinstance(final_cost,(final_SOCcontraint)),"Final cost must come from RMC.costfunctions"
        assert batch_size>=0, "Batch size cannot be negative."
        assert nsim >0,"Number of simulations cannot be negative"

        super(HybridControl,self).__init__(nsim,underlying_process,running_cost,final_cost,BESSparameters,\
                                           batch_size,value_kernel,normalize_value,policy_kernel,normalize_policy)

    @property
    def save_policy_maps(self):
        os.makedirs('PolicyMaps',exist_ok = True)

        for i, model in enumerate(self.policy_maps):
            filename = f'model{i}.pkl'
            filepath = os.path.join('PolicyMaps',filename)
            with open(filepath,'wb') as file:
                pickle.dump(model,file)
    
    @property
    def save_value_maps(self):
        os.makedirs('ValueMaps',exist_ok = True)

        for i, model in enumerate(self.value_maps):
            filename = f'model{i}.pkl'
            filepath = os.path.join('ValueMaps',filename)
            with open(filepath,'wb') as file:
                pickle.dump(model,file)