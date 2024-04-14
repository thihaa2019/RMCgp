import numpy as np


class L2():
    @staticmethod
    def cost(B,X,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"
        return ((B-X+target)**2) 

    @staticmethod
    def derivative(B,X,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"
        return (2*(B-X+target)) 
    

class L1():
    @staticmethod
    def cost(B,X,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"

        return np.abs(B-X+target)

    @staticmethod
    def derivative(B,X,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"
        return (1* (B-X+target>0) - 1* (B-X+target<0) )
    
class penalized_L2():
    def __init__(self):
        self._upper_pen = None
        self._lower_pen = None

    @property
    def upper_pen(self):
        return self._upper_p
    @upper_pen.setter
    def upper_pen(self,P1):
        self._upper_pen = P1
    
    @property
    def lower_pen(self):
        return self._lower_pen
    @lower_pen.setter
    def lower_pen(self,P2):
        self._lower_pen = P2

    def cost(self,B,X,*args):
        target,lower_target,upper_target = args

        cost1  = self._upper_pen*(X-B - upper_target)**2 * (X-B>upper_target)
        cost2 = self._lower_pen*(lower_target- (X- B))**2 * (lower_target> X-B)
        #cost3 = (B[0]-X+target_lvl)**2 * (X-B[0] >lower_target) *  (X-B[0] <upper_target)
        cost3 = (B-X+target)**2
        total_cost = cost1+cost2+cost3
        return total_cost
    
    def derivative(self,B,X,*args):

        target,lower_target,upper_target = args
        cost1_derivative = -2*self._upper_pen *(X-B - upper_target)* (X-B>upper_target)
        #cost2_derivative =    2 * (lower_target- (X- B[0])) * (lower_target> X-B[0])
        cost2_derivative = 2*self._lower_pen*(lower_target- (X- B))*(lower_target> X-B)
        #cost3_derivative =  2* (B[0]-X+target_lvl)* (X-B[0] >lower_target) *  (X-B[0] <upper_target)
        cost3_derivative = 2* (B-X+target)
        current_derivative = cost1_derivative+cost2_derivative+cost3_derivative
        return current_derivative