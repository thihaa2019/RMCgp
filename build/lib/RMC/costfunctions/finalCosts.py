import numpy as np

class final_SOCcontraint():
    def __init__(self,percent_level,scale):
        self.constraint_level = percent_level
        self.scale = scale
        self._Imax = None
        self._charging_eff = None
    
    @property
    def Imax(self):
        return self._Imax
    @Imax.setter
    def Imax(self,imax):
        self._Imax = imax

    @property
    def charging_eff(self):
        return self._charging_eff
    @charging_eff.setter
    def charging_eff(self,efficiency):
        self._charging_eff = efficiency

    def cost(self,I):
        return self.scale*np.maximum(self._Imax*self.constraint_level- I,0)
    
    def derivative(self,I,B):
        # assume I follows I_prev + eta B( B>0) dt + 1/eta B(B<0 ) dt
        dg_dB = -self.scale * (I< self._Imax*self.constraint_level ) * \
            (self._charging_eff * (B>0) + 1/self._charging_eff * (B<0))
        return dg_dB