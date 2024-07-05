import numpy as np


class L2():
    @staticmethod
    def cost(B,X,I,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"
        return ((B-X+target)**2) 

    @staticmethod
    def derivative(B,X,I,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"
        return (2*(B-X+target)) 

class L2_degradation1():
    def __init__(self,pen):
        self.degradation_coef = pen

    def cost(self,B,X,I,*args):
        target = args[0]
        Imax = 8
        running_cost = (B-X+target)**2 
        c_1  = 960
        degradation_cost =  self.degradation_coef * ((1-0.5*(I/Imax)**2) * (-B) * (B<0) *(1/c_1))
        return running_cost+degradation_cost

    def derivative(self,B,X,I,*args):
        target = args[0]
        running_derivative = 2* (B-X+target)
        Imax = 8
        c_1  = 960
        degradation_derivative = self.degradation_coef * ((1-0.5*(I/Imax)**2) * (-1) * (B<0) *(1/c_1))
        return running_derivative+degradation_derivative

class L2_degradation2():
    def __init__(self,pen):
        self.degradation_coef = pen

    def cost(self,B,X,I,*args):
        target = args[0]
        Imax = 8
        running_cost = (B-X+target)**2 
        c_1  = 960*2
        degradation_cost1 = self.degradation_coef * ((1-0.5*(I/Imax)**2) *  (-B) * (B<0) * (1/c_1)) 
        degradation_cost2 = self.degradation_coef * ( (1-0.5*(1- I/Imax)**2) * (B) * (B>0) * (1/c_1) )
        return running_cost+degradation_cost1+degradation_cost2

    def derivative(self,B,X,I,*args):
        target = args[0]
        running_derivative = 2* (B-X+target)
        Imax = 8
        c_1  = 960*2
        degradation_derivative1 = self.degradation_coef * ((1-0.5*(I/Imax)**2) *  (-1) * (B<0) * (1/c_1)) 
        degradation_derivative2 =  self.degradation_coef * ( (1-0.5*(1- I/Imax)**2) * (1) * (B>0) * (1/c_1) )
        return running_derivative+degradation_derivative1+degradation_derivative2

class L2_degradation3():
    def __init__(self,pen):
        self.degradation_coef = pen

    def cost(self,B,X,I,*args):
        target = args[0]
        Imax = 8
        running_cost = (B-X+target)**2 
        c1  = 15;c2 = 1e5;c3 = 1e-4 *0.6
        degradation_cost =self.degradation_coef * ( (c1*(I/Imax - 0.5))**2 * B**2/c2 + c3 * B**2)
        return running_cost+degradation_cost

    def derivative(self,B,X,I,*args):
        target = args[0]
        running_derivative = 2* (B-X+target)
        Imax = 8
        c1  = 15;c2 = 1e5;c3 = 1e-4 *0.6
        degradation_derivative= self.degradation_coef * ((c1*(I/Imax - 0.5))**2 * (2*B) /c2 + c3 * (2*B))
        return running_derivative+degradation_derivative

class L1():
    @staticmethod
    def cost(B,X,I,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"
        return np.abs(B-X+target)

    @staticmethod
    def derivative(B,X,I,*args):
        target = args[0]
        assert isinstance(target, (int,float)), "Target must be scalar"
        return (1* (B-X+target>0) - 1* (B-X+target<0) )

class penalized_L2():
    def __init__(self,lower_pen,mid_pen,upper_pen):
        self._upper_pen = upper_pen
        self._lower_pen = lower_pen
        self._mid_pen   = mid_pen


    def cost(self,B,X,I,*args):
        target,lower_target,upper_target = args

        cost1  = self._upper_pen*(X-B - upper_target)**2 * (X-B>upper_target)
        cost2 = self._lower_pen*(lower_target- (X- B))**2 * (lower_target> X-B)
        #cost3 = (B[0]-X+target_lvl)**2 * (X-B[0] >lower_target) *  (X-B[0] <upper_target)
        cost3 = self._mid_pen*(B-X+target)**2
        total_cost = cost1+cost2+cost3
        return total_cost
    
    def derivative(self,B,X,I,*args):

        target,lower_target,upper_target = args
        cost1_derivative = -2*self._upper_pen *(X-B - upper_target)* (X-B>upper_target)
        #cost2_derivative =    2 * (lower_target- (X- B[0])) * (lower_target> X-B[0])
        cost2_derivative = 2*self._lower_pen*(lower_target- (X- B))*(lower_target> X-B)
        #cost3_derivative =  2* (B[0]-X+target_lvl)* (X-B[0] >lower_target) *  (X-B[0] <upper_target)
        cost3_derivative = 2*self._mid_pen*  (B-X+target)
        current_derivative = cost1_derivative+cost2_derivative+cost3_derivative
        return current_derivative
    

