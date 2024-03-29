import numpy as np


class L2():
    @staticmethod
    def cost(B,X,target):
        assert isinstance(target, (int,float)), "Target must be scalar"
        return ((B-X+target)**2) 

    @staticmethod
    def derivative(B,X,target):
        assert isinstance(target, (int,float)), "Target must be scalar"
        return (2*(B-X+target)) 
    

class L1():
    @staticmethod
    def cost(B,X,target):
        assert isinstance(target, (int,float)), "Target must be scalar"

        return np.abs(B-X+target)

    @staticmethod
    def derivative(B,X,target):
        assert isinstance(target, (int,float)), "Target must be scalar"
        return (1* (B-X+target>0) - 1* (B-X+target<0) )