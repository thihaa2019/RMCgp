<h1 align='center'> An RMC library for renewable stochastic control problems 
    [<a href="https://arxiv.org/">arXiv</a>] </h1>

<p align="center">
<img align="middle" src="../imgs/L1vL2trajectory.png" width="666" />
</p>

Building on the well-understood mathematical theory of _stochstic optimal control_, we solve renewable energy control problems using RMC that involves:
+ Dynamic Programming
+ Policy and Value emulators
+ Demonstrate state-of-the-art performance.

The method is straight forward to implement and evaluate using existing tools, in particular GP and the [`GPy`](https://github.com/SheffieldML/GPy) library.

----

## Install

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
See GitHub repository [GitHub](https://github.com/thihaa2019/RMCgp/), which demonstrates how to use the library to train an optimal control model for hybrid renewabl-battery asset.



### Citation

```
