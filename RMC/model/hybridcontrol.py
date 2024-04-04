from .hybridrunner import runRMC
from ..simulate import Sim
from ..costfunctions import L1,L2,final_SOCcontraint
from .test_inputs import cost_tester,kernel_tester
import GPy
import os
import pickle
class HybridControl(runRMC):
    def __init__(self,nsim,underlying_process,running_cost,final_cost,BESSparameters,batch_size,value_kernel,normalize_value,policy_kernel,normalize_policy,*args):
        

        #hasattr(RMC.costfunctions.L1,"cost") and callable(RMC.costfunctions.L1.cost)
        assert isinstance(underlying_process,Sim), "Underlying must come from RMC.sim"
        assert len(BESSparameters) ==4, "Bmax,Imax,charging efficiency, SOC limits must be defined"
        assert batch_size>=0, "Batch size cannot be negative."
        assert nsim >0,"Number of simulations cannot be negative"

        cost_tester(running_cost)
        cost_tester(final_cost)
        kernel_tester(value_kernel)
        kernel_tester(policy_kernel)

        super(HybridControl,self).__init__(nsim,underlying_process,running_cost,final_cost,BESSparameters,\
                                           batch_size,value_kernel,normalize_value,policy_kernel,normalize_policy,*args)


    def save_policy_maps(self,folder_name):
        os.makedirs(folder_name,exist_ok = True)

        for i, model in enumerate(self.policy_maps):
            filename = f'model{i}.pkl'
            filepath = os.path.join(folder_name,filename)
            with open(filepath,'wb') as file:
                pickle.dump(model,file)

    def save_value_maps(self,folder_name):
        os.makedirs(folder_name,exist_ok = True)

        for i, model in enumerate(self.value_maps):
            filename = f'model{i}.pkl'
            filepath = os.path.join(folder_name,filename)
            with open(filepath,'wb') as file:
                pickle.dump(model,file)