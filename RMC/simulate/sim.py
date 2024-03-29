import numpy as np

class Sim():

    """
    drift is a function of 3 parameters p1scalar, p2vector(t), p3X(t)
    
    
    """
    def __init__(self,X0,nstep,nsim,maturity,drift_parameters,drift_function,diffusion_parameters,diffusion_function):

        if isinstance(X0,(list,np.ndarray)):
            assert len(X0)==nsim, "X0 and nsim must have same size"
        self.X0 = X0

        assert maturity>0, "Maturity cannot be 0 or less."
        assert nstep>0, "Number of steps cannot be 0 or less."
        self.maturity = maturity
        self.nstep    = nstep
        self.dt = maturity/nstep
        self.t = np.linspace(0,maturity,nstep+1)
        if nsim==None:
            self.nsim = 10000
        else:
            self.nsim = nsim

        self.drift = drift_function
        assert isinstance(drift_parameters[0],(np.ndarray,list)) and isinstance(drift_parameters[1],(np.ndarray,list)),\
            "Drift Parameters must be array"
        self.drift_p1 = drift_parameters[0]
        self.drift_p2 = drift_parameters[1]
        if len(self.drift_p1)==1:
            self.drift_p1 = list(self.drift_p1)* self.nstep
        if len(self.drift_p2)==1:
            self.drift_p2 = list(self.drift_p2)* self.nstep

        self.diffusion = diffusion_function
        assert isinstance(diffusion_parameters[0],(np.ndarray,list)) and isinstance(diffusion_parameters[1],(np.ndarray,list)),\
            "Diffusion Parameters must be array"
        self.diffusion_p1 = diffusion_parameters[0]
        self.diffusion_p2 = diffusion_parameters[1]
        if len(self.diffusion_p1)==1:
            self.diffusion_p1 = list(self.diffusion_p1)* self.nstep
        if len(self.diffusion_p2)==1:
            self.diffusion_p2 = list(self.diffusion_p2)* self.nstep

        self.sim_trajectories = self.simulate()
    
    def simulate(self,BM = None,given_BM = False):
        self.Xs = np.zeros((self.nsim,self.nstep+1))
        self.Xs[:,0] = np.ones(self.nsim)* self.X0

        if not given_BM:
            dW = np.random.normal(0,1,size = (self.nsim,self.nstep) ) * np.sqrt(self.dt)
        else:
            dW = BM

        for i in range(1,self.nstep+1):
            self.Xs[:,i] = np.abs(self.Xs[:,i-1] +self.drift(self.drift_p1[i-1],self.drift_p2[i-1],self.Xs[:,i-1]) * self.dt +\
                            self.diffusion(self.diffusion_p1[i-1],self.diffusion_p2[i-1],self.Xs[:,i-1])* dW[:,i-1])
        return self.Xs
    @property
    def mean_vec(self):
        return np.mean(self.sim_trajectories,axis = 0)
    @property
    def std_vec(self):
        return np.std(self.sim_trajectories,axis= 0, ddof = 1)

    def onestepsimulate(self,nsim,X_start,drift_p1,drift_p2,vol_p1,vol_p2):
        dW = np.random.normal(0,1,size = nsim) * np.sqrt(self.dt)
        X_next = np.abs(X_start +self.drift(drift_p1,drift_p2,X_start) * self.dt +\
                            self.diffusion(vol_p1,vol_p2,X_start)* dW)
        return X_next
    def cor(self,i,j):
        if i <=0 or i >= self.nstep+1:
            raise ValueError("Invalid index for i.")
        if j <= 0 or j >= self.nstep+1:
            raise ValueError("Invalid index for j.")
        
        return np.corrcoef(self.sim_trajectories[:,i],self.sim_trajectories[:,j])[0,1]
    @property
    def CI_95(self):

        lower,upper = np.quantile(self.sim_trajectories, q =[0.025,0.975],axis = 0)
        return lower,upper


        