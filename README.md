<h1 align='center'> An RMC library for renewable stochastic control problems 


### Introduction 

Building on the well-understood mathematical theory of _stochstic optimal control_, we solve renewable energy control problems using RMC that involves:
+ Dynamic Programming
+ Policy and Value emulators
+ Demonstrate state-of-the-art performance.

The method is straight forward to implement and evaluate using existing tools, in particular GP and the [`GPy`](https://github.com/SheffieldML/GPy) library.

----

## Getting started: installing with pip

We strongly recommend using
the  [anaconda python distribution](http://continuum.io/downloads).
With anaconda you can install RMCpy by the following:


    sudo apt-get update
    sudo apt-get install python3-dev
    sudo apt-get install build-essential   
    conda update anaconda
    
And finally,

    pip install RMCgp

### Example
We encourage looking at [nb.ipynb](https://github.com/thihaa2019/RMCgp/blob/main/nb.ipynb), which demonstrates how to use the library to train an optimal control model for hybrid renewabl-battery asset.

A self contained short example:
```python
import RMC
import numpy as np
import GPy
### Defining Model 
X0 =np.random.normal(5,np.sqrt(0.5),10000)
process = RMC.simulate.OU(X0,96,10000,24,[1],[5],[1])
running_cost = RMC.costfunctions.L2()
final_cost = RMC.costfunctions.final_SOCcontraint(0,0)
parameters = (2,8,0.95,0.05)
batch_size = 30
value_kernel= GPy.kern.Matern52
normalize_v = True
policy_kernel = GPy.kern.Matern32
normalize_policy = True
hybrid_solution = RMC.model.HybridControl(600,process,running_cost,final_cost,parameters,batch_size,\
                                          value_kernel,normalize_v,policy_kernel,normalize_policy)

## Run RMC solve
hybrid_solution.solve()
```


### Reproducing experiments

### Results




### Citation

```
