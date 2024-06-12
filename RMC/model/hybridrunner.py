
from ..optimization import onestepOptimizer
from ..design import design_generator
from .. emulator import valueGP,policyGP
from .. costfunctions import final_SOCcontraint,quadratic_SoC_constraint
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
        self.X_lowers = np.ones(len(self.X_lowers))*np.min(self.process.sim_trajectories)
        self.X_uppers = np.ones(len(self.X_uppers))*np.max(self.process.sim_trajectories)
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

        self.MSE_evaluation(control_output.flatten(),control_map.predict(control_input)[0].flatten(),"Policy",I_train)

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
        target,lower_target,upper_target,step = args
        assert isinstance(target,(int,float)), "One step target is a scalar"
        reshape_dim = 1 # no reshaping
        X_prev_rep = X_prev
        I_next_rep = I_next
        if self.batch_size!=0:  
            reshape_dim = self.batch_size
            X_prev_rep= np.repeat(X_prev,self.batch_size)
            I_next_rep= np.repeat(I_next,self.batch_size)

        X_next_rep= self.process.onestepsimulate(int(self.nsim*reshape_dim),X_prev_rep,drift_p1,drift_p2,vol_p1,vol_p2,step)
        power_outputs = policy_map.predict(np.column_stack((X_next_rep,I_next_rep)))[0].flatten()
        LB = np.maximum(self.Bmin,self.charging_eff*(self.Ilb- I_next_rep)/self.dt)
        UB = np.minimum(self.Bmax,(self.Iub-I_next_rep)/(self.charging_eff*self.dt))
        # according to scipy.optimize, B>0 when X>=M
        #pos_outputs = np.where(power_outputs>0)
        # do not charge more than X-M, causing O < M , ensures B<= X-M or M<=X-B
        #upper_charging_bound  = np.maximum(X_next_rep[pos_outputs]-target,0)
       #power_outputs[pos_outputs] = np.minimum(power_outputs[pos_outputs],upper_charging_bound)

        # according to scipy.optimize, B<0 when X<=M
        #neg_outputs = np.where(power_outputs<0)
        # do not discharge more than X-M, causing O > M , ensures B>= X-M or M>=X-B
        #lower_charging_bound  = np.minimum(X_next_rep[neg_outputs]-target,0)
        #power_outputs[neg_outputs] = np.maximum(power_outputs[neg_outputs],lower_charging_bound)

        power_outputs = np.maximum(LB,np.minimum(power_outputs,UB))
        I_nextnext = I_next_rep + power_outputs *(self.charging_eff *(power_outputs>0) + 1/self.charging_eff * (power_outputs<0))*self.dt
        if isinstance(value_map,final_SOCcontraint) or isinstance(value_map,quadratic_SoC_constraint):
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
                                              self.process.diffusion_p1[-2],self.process.diffusion_p2[-2],-2)
        self.policies[-1] = self.train_policy(X_next,I_next,self.values[-1],self.targets[-1],self.lower_targets[-1],self.upper_targets[-1])

        
        pointwise_values = self.v_evaluation(X_prev,self.process.drift_p1[-2],self.process.drift_p2[-2],\
                                             self.process.diffusion_p1[-2],self.process.diffusion_p2[-2],\
                                             I_next,self.policies[-1],self.values[-1],\
                                                self.targets[-1],self.lower_targets[-1],self.upper_targets[-1],-2)
        
        for iStep in range(self.nstep-1,0,-1):
            self.values[iStep-1] = self.train_value(X_prev,I_next,pointwise_values)

            print("Timestep: "+ str(self.t[iStep]))

            print("_"*50)

            if iStep!=1:
                X_prev,I_next = design_generator(self.nsim,self.X_lowers[iStep-2],self.X_uppers[iStep-2],self.Ilb,self.Iub).create_samples
                print(X_prev.shape)
                X_next = self.process.onestepsimulate(self.nsim,X_prev,self.process.drift_p1[iStep-2],self.process.drift_p2[iStep-2],\
                                                self.process.diffusion_p1[iStep-2],self.process.diffusion_p2[iStep-2],iStep-2)

                self.policies[iStep-1] = self.train_policy(X_next,I_next,self.values[iStep-1],self.targets[iStep-1],self.lower_targets[iStep-1],self.upper_targets[iStep-1])

        
                pointwise_values = self.v_evaluation(X_prev,self.process.drift_p1[iStep-2],self.process.drift_p2[iStep-2],\
                                             self.process.diffusion_p1[iStep-2],self.process.diffusion_p2[iStep-2],\
                                             I_next,self.policies[iStep-1],self.values[iStep-1],\
                                            self.targets[iStep-1],self.lower_targets[iStep-1],self.upper_targets[iStep-1],iStep-2)
        
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
    

    def onePath_control(self,A_t,I0):
        X_vec = A_t
        I_vec = np.zeros(len(X_vec)+1)
        assert I0<= self.Iub and I0>=self.Ilb, "I not in bound"
        I_vec[0] =I0
        B_vec = np.zeros(len(X_vec))
        total_cost = 0
        for i in range(len(A_t)):
            cur_X =  X_vec[i]
            cur_I  = I_vec[i]
            inp_vec = np.array([cur_X,cur_I]).reshape(1,-1)
            B = self.policy_maps[i].predict(inp_vec)[0].flatten()

            #if B >0:
            #    B= np.minimum(B,cur_X-self.targets[i])
            #if B<0:
            #    B = np.maximum(cur_X-self.targets[i],B)
            LB = np.maximum(self.Bmin,self.charging_eff*(self.Ilb- cur_I)/self.dt)
            UB = np.minimum(self.Bmax,(self.Iub-cur_I)/(self.charging_eff*self.dt))
            B = np.maximum(LB,np.minimum(B,UB))

            B_vec[i] = B
            cur_I = cur_I + B * ( (B>0) * self.charging_eff + (B<0) *1/self.charging_eff) * self.dt
            total_cost = total_cost+ self.running_cost.cost(B,cur_X,self.targets[i],self.lower_targets[i],self.upper_targets[i])* self.dt
            I_vec[i+1]=cur_I
        total_cost += self.final_cost.cost(I_vec[-1])
        return X_vec,I_vec,B_vec,total_cost

    def monteCarlo_GPcontrol(self,X0,I0,N_MC,init_sigma = None, randomize = False,new_sim =False):
        if randomize:
            assert init_sigma>0, "Sigma must be positive"
            X_init = np.minimum(np.random.normal(X0,init_sigma,size = (N_MC) ),self.process.UB)
            X_init = np.maximum(X_init,self.process.LB)
        self.process.X0 = X0
        self.process.nsim = N_MC
        X_sims = self.process.simulate(new_sim)
        Bts = np.zeros((N_MC,self.nstep))
        Is = np.zeros((N_MC,self.nstep+1)); Is[:,0] = I0 *np.ones(N_MC)

        running_cost = np.zeros((N_MC,self.nstep))

        for i in range(self.nstep):
            LB = np.maximum(self.Bmin,self.charging_eff*(self.Ilb- Is[:,i])/self.dt)
            UB = np.minimum(self.Bmax,(self.Iub-Is[:,i])/(self.charging_eff*self.dt))
            inp = np.column_stack((X_sims[:,i],Is[:,i]))
            Bts[:,i] = self.policy_maps[i].predict(inp)[0].flatten()
            # according to scipy.optimize, B>0 when X>=M
            #pos_outputs = np.where(Bts[:,i] >0)
            # do not charge more than X-M, causing O < M , ensures B<= X-M or M<=X-B
            #upper_charging_bound  = np.maximum(X_sims[pos_outputs,i]-self.targets[i],0)
            #Bts[pos_outputs,i] = np.minimum(Bts[pos_outputs,i],upper_charging_bound)

            # according to scipy.optimize, B<0 when X<=M
            #neg_outputs = np.where(Bts[:,i] <0)
            # do not discharge more than X-M, causing O > M , ensures B>= X-M or M>=X-B
            #lower_charging_bound  = np.minimum(X_sims[neg_outputs,i]-self.targets[i],0)
            #Bts[neg_outputs,i]= np.maximum(Bts[neg_outputs,i],lower_charging_bound)
            Bts[:,i] = np.maximum(LB,np.minimum(Bts[:,i],UB))
            Is[:,i+1] = Is[:,i]+Bts[:,i]*(self.charging_eff*(Bts[:,i]>0) +1/self.charging_eff *(Bts[:,i]<0))*self.dt
            running_cost[:,i] = self.running_cost.cost(Bts[:,i],X_sims[:,i],self.targets[i],self.lower_targets[i],self.upper_targets[i])* self.dt
        total_cost = np.sum(running_cost,axis = 1) +self.final_cost.cost(Is[:,-1]).flatten()
        mean_total_cost = np.mean(total_cost)

        return X_sims,Is,Bts, mean_total_cost


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
        
