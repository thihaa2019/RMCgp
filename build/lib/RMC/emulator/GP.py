import GPy
import pickle

class valueGP(GPy.core.GP):

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):

        if kernel is None:
            kernel = GPy.kern.Matern52(input_dim=X.shape[1],ARD= True)
        assert isinstance(kernel, GPy.kern.Kern)
        likelihood = GPy.likelihoods.Gaussian(variance=noise_var)
        super(valueGP, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

    def save_model(self,output_filename):
       with open(output_filename+'.pkl', 'wb') as file:
            pickle.dump(self, file)

class policyGP(GPy.core.GP):

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):
        if kernel is None:
            kernel = GPy.kern.Matern32(input_dim=X.shape[1],ARD= True)
        assert isinstance(kernel, GPy.kern.Kern), "Kernel must be GPy kernel."
        likelihood = GPy.likelihoods.Gaussian(variance=noise_var)

        super(policyGP, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

    def save_model(self,output_filename):
       with open(output_filename+'.pkl', 'wb') as file:
            pickle.dump(self, file)
            
