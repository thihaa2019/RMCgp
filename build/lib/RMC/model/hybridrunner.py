
from ..optimization import onestepOptimizer
from ..design import design_generator
from .. emulator import valueGP,policyGP
from .. costfunctions import final_SOCcontraint
import GPy
import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
class runRMC():
    def __init__(self,nsim,underlying_process,running_cost,final_cost,BESSparameters,batch_size,value_kernel,normalize_value,policy_kernel,normalize_policy,*args):
        self.nsim = nsim
        self.process = underlying_process
        self.nstep = self.process.nstep
        self.dt = self.process.dt
        self.t  = self.process.t
        self.batch_size = batch_size
        self.running_cost = running_cost

        self.Bmax,self.Imax,self.charging_eff,self.SOC_percen_lim = BESSparameters
        self.Bmin = -self.Bmax
        self.Iub = (1-self.SOC_percen_lim)* self.Imax
        self.Ilb = self.SOC_percen_lim * self.Imax
        
        self.final_cost   = final_cost
        self.final_cost.Imax = self.Imax
        self.final_cost.charging_eff = self.charging_eff
        
        self.kernel_dic = {"RBF": GPy.kern.RBF, "Matern52":GPy.kern.Matern52 , "Matern32":GPy.kern.Matern32, None:None}
        self.value_kernel = self.kernel_dic[value_kernel]
        self.policy_kernel= self.kernel_dic[policy_kernel]
        self.value_normalizer = normalize_value
        self.policy_normalizer = normalize_policy


        self.values = [None]*self.nstep
        self.policies = [None]*self.nstep
        self.optimizer = onestepOptimizer(self.running_cost,self.charging_eff,self.dt)
        # final cost parameters are fixed
        # design parameters keep chanding -> dynamic
        # onestep optimizer initialization fixed, but reset q at each step, then set b,x,i, target
        
        self.targets = self.process.drift_p2
        self.lower_targets = [None]* self.nstep
        self.upper_targets = [None]*self.nstep
        if len(args)==2:
            self.lower_targets,self.upper_targets = args
        assert len(self.targets) == self.nstep, "Target length must be nstep total."
        assert len(self.lower_targets) == self.nstep, "Lower length must be nstep total."
        assert len(self.upper_targets) == self.nstep, "Target length must be nstep total."


        self.X_lowers,self.X_uppers = np.quantile(self.process.sim_trajectories, q =[0.005,0.995],axis = 0)
    def train_policy(self,X_train,I_train,value_map,*args):
        assert len(X_train)==len(I_train)
        B_train = np.zeros(len(X_train))

        for i in range(len(B_train)):
            B_train[i] = self.optimizer.optimize(X_train[i],I_train[i],value_map,*args)
        control_input = np.column_stack((X_train,I_train))
        control_output= B_train.reshape(-1,1)
        if self.policy_kernel is None:
            kern = None
        else:
            kern = self.policy_kernel(control_input.shape[1],ARD = True)
        

        control_map = policyGP(control_input,control_output,kernel =kern,normalizer= True)
        #control_map.optimize_restarts(num_restarts=2)
        control_map.optimize()

        MSE = self.MSE_evaluation(control_output.flatten(),control_map.predict(control_input)[0].flatten(),"Policy",I_train)
        while MSE>=1e-1:
            X_lb,X_ub = np.min(X_train),np.max(X_train)
            X_new,I_new = design_generator(50,X_lb,X_ub,self.Ilb,self.Iub).create_samples
            new_control_input = np.column_stack((X_new,I_new))
            new_control_output = np.zeros(50)
            for i in range(len(new_control_output)):
                new_control_output[i] = self.optimizer.optimize(X_new[i],I_new[i],value_map,*args)
            new_control_output = new_control_output.reshape(-1,1)
            control_map.set_XY(np.vstack((control_map.X, new_control_input)), np.vstack((control_map.Y, new_control_output)))
            control_map.optimize_restarts(num_restarts=2)
            control_map.optimize()
            MSE = self.MSE_evaluation(control_map.Y.flatten(),control_map.predict(control_map.X)[0].flatten(),\
                                        "Policy",control_map.X[:,1])
        #print(len(control_map.optimization_runs))
        return control_map

    def train_value(self,X_train,I_train,pointwisevalues):
        value_input = np.column_stack((X_train,I_train))
        value_output= pointwisevalues.reshape(-1,1)
        if self.value_kernel is None:
            kern = None
        else:
            kern = self.value_kernel(value_input.shape[1],ARD = True)

        value_map = valueGP(value_input,value_output,kernel= kern,normalizer=True)
        value_map.optimize()
        self.MSE_evaluation(value_output.flatten(),value_map.predict(value_input)[0].flatten(),"Value")
        return value_map
    
    def v_evaluation(self,X_prev,drift_p1,drift_p2,vol_p1,vol_p2,I_next,policy_map,value_map,*args):
        target,lower_target,upper_target = args
        assert isinstance(target,(int,float)), "One step target is a scalar"
        reshape_dim = 1 # no reshaping
        X_prev_rep = X_prev
        I_next_rep = I_next
        if self.batch_size!=0:  
            reshape_dim = self.batch_size
            X_prev_rep= np.repeat(X_prev,self.batch_size)
            I_next_rep= np.repeat(I_next,self.batch_size)

        X_next_rep= self.process.onestepsimulate(int(self.nsim*reshape_dim),X_prev_rep,drift_p1,drift_p2,vol_p1,vol_p2)
        power_outputs = policy_map.predict(np.column_stack((X_next_rep,I_next_rep)))[0].flatten()
        LB = np.maximum(self.Bmin,self.charging_eff*(self.Ilb- I_next_rep)/self.dt)
        UB = np.minimum(self.Bmax,(self.Iub-I_next_rep)/(self.charging_eff*self.dt))

        pos_outputs = np.where(power_outputs>0)
        # cannot charge more than X is available
        upper_charging_bound  = np.maximum(X_next_rep[pos_outputs]-target,0)
        power_outputs[pos_outputs] = np.minimum(power_outputs[pos_outputs],upper_charging_bound)
        power_outputs = np.maximum(LB,np.minimum(power_outputs,UB))
        I_nextnext = I_next_rep + power_outputs *(self.charging_eff *(power_outputs>0) + 1/self.charging_eff * (power_outputs<0))*self.dt
        if isinstance(value_map,final_SOCcontraint):
            pointwise_v = self.running_cost.cost(power_outputs,X_next_rep,target,lower_target,upper_target) * self.dt + value_map.cost(I_nextnext)
        if isinstance(value_map,GPy.core.GP):
            inp = np.column_stack((X_next_rep,I_nextnext))
            pointwise_v = self.running_cost.cost(power_outputs,X_next_rep,target,lower_target,upper_target) * self.dt + value_map.predict(inp)[0].flatten()
        pointwise_v = np.mean(pointwise_v.reshape(-1,reshape_dim),axis = 1)

        return pointwise_v
    
    def solve(self):


        # need to be fixed
        #self.X_lowers[0] = self.process.mean_vec[0] - 2.5*self.process.std_vec[3]
        #self.X_uppers[0] = self.process.mean_vec[0] + 2.5*self.process.std_vec[3]

        self.values[-1] = self.final_cost 
        X_prev,I_next = design_generator(self.nsim,self.X_lowers[-3],self.X_uppers[-3],self.Ilb,self.Iub).create_samples
        X_next = self.process.onestepsimulate(self.nsim,X_prev,self.process.drift_p1[-2],self.process.drift_p2[-2],\
                                              self.process.diffusion_p1[-2],self.process.diffusion_p2[-2])
        self.policies[-1] = self.train_policy(X_next,I_next,self.values[-1],self.targets[-1],self.lower_targets[-1],self.upper_targets[-1])

        
        pointwise_values = self.v_evaluation(X_prev,self.process.drift_p1[-2],self.process.drift_p2[-2],\
                                             self.process.diffusion_p1[-2],self.process.diffusion_p2[-2],\
                                             I_next,self.policies[-1],self.values[-1],\
                                                self.targets[-1],self.lower_targets[-1],self.upper_targets[-1])
        
        for iStep in range(self.nstep-1,0,-1):
            self.values[iStep-1] = self.train_value(X_prev,I_next,pointwise_values)

            print("Timestep: "+ str(self.t[iStep]))

            print("_"*50)

            if iStep!=1:
                X_prev,I_next = design_generator(self.nsim,self.X_lowers[iStep-2],self.X_uppers[iStep-2],self.Ilb,self.Iub).create_samples
                X_next = self.process.onestepsimulate(self.nsim,X_prev,self.process.drift_p1[iStep-2],self.process.drift_p2[iStep-2],\
                                                self.process.diffusion_p1[iStep-2],self.process.diffusion_p2[iStep-2])

                self.policies[iStep-1] = self.train_policy(X_next,I_next,self.values[iStep-1],self.targets[iStep-1],self.lower_targets[iStep-1],self.upper_targets[iStep-1])

        
                pointwise_values = self.v_evaluation(X_prev,self.process.drift_p1[iStep-2],self.process.drift_p2[iStep-2],\
                                             self.process.diffusion_p1[iStep-2],self.process.diffusion_p2[iStep-2],\
                                             I_next,self.policies[iStep-1],self.values[iStep-1],\
                                            self.targets[iStep-1],self.lower_targets[iStep-1],self.upper_targets[iStep-1])
        
            else:
                X_0,I_0 = design_generator(self.nsim,self.X_lowers[0],self.X_uppers[0],self.Ilb,self.Iub).create_samples
                #self.optimizer.q = self.values[iStep-1]
                self.policies[iStep-1] = self.train_policy(X_0,I_0,self.values[iStep-1],self.targets[iStep-1],self.lower_targets[iStep-1],self.upper_targets[iStep-1])
                print("Timestep: "+ str(self.t[iStep-1]))
                print("_"*50)

    @property
    def policy_maps(self):
        return self.policies
    
    @property
    def value_maps(self):
        return self.values
    
    def MSE_evaluation(self,y_true,GP_output,map_type,*args):
        if map_type == "Policy":
            cur_I = args[0]
            LB = np.maximum(self.Bmin,self.charging_eff*(self.Ilb- cur_I)/self.dt)
            UB = np.minimum(self.Bmax,(self.Iub-cur_I)/(self.charging_eff*self.dt))
            p1 = np.maximum(LB,np.minimum(y_true,UB))
            p2 = np.maximum(LB,np.minimum(GP_output,UB))
            MSE = np.mean((p1-p2)**2)
        else:
            MSE = np.mean((y_true - GP_output)**2)
        print(f"{map_type} GP MSE: " + str(MSE))
        return MSE
        
