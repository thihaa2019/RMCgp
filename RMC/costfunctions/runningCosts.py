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

    
class firming_L2():
    def __init__(self,weight1,weight2 ):
        self.w1 = weight1
        self.w2 = weight2
    def cost(self,B,X,I,*args):
        target,_,upper_target= args
        L2cost  = self.w1* (X-B - target)**2

        #lower_soft = self.w1* (lower_target/2- (X- B))**2 * (lower_target/2> X-B)
        #upper_soft = self.w1* (X-B - upper_target/2)**2 * (X-B>upper_target/2)

        upper_hard = self.w2 *(X-B-upper_target)**2 * (X-B>upper_target)
        total_cost = L2cost +  upper_hard

        return total_cost
    def derivative(self,B,X,I,*args):
        target,_,upper_target = args
        L2derivative = -2 *self.w1* (X-B - target)

        #lower_soft = 2 * self.w1* (lower_target/2- (X- B)) * (lower_target/2> X-B)
        #upper_soft = -2* self.w1* (X-B - upper_target/2)   * (X-B>upper_target/2)

        upper_hard = -2* self.w2 *(X-B-upper_target)    * (X-B>upper_target)
        total_derivative = L2derivative+   upper_hard
        return total_derivative
    
class penalty_L2():
    def __init__(self,weight1,weight2 ):
        self.w1 = weight1
        self.w2 = weight2
    def cost(self,B,X,I,*args):
        target,_,upper_target= args
        L2cost  = self.w1* (X-B - target)**2

        #lower_soft = self.w1* (lower_target/2- (X- B))**2 * (lower_target/2> X-B)
        #upper_soft = self.w1* (X-B - upper_target/2)**2 * (X-B>upper_target/2)

        upper_hard = self.w2 *np.maximum(X-B- upper_target,0)
        total_cost = L2cost + upper_hard
        return total_cost

    def derivative(self,B,X,I,*args):
        target,_,upper_target = args
        L2derivative = -2 *self.w1* (X-B - target)

        #lower_soft = 2 * self.w1* (lower_target/2- (X- B)) * (lower_target/2> X-B)
        #upper_soft = -2* self.w1* (X-B - upper_target/2)   * (X-B>upper_target/2)

        arg = X - B - upper_target
        upper_hard = -self.w2* (arg>0)
        total_derivative = L2derivative+  upper_hard
        return total_derivative
    

    