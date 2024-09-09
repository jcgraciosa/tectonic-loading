import numpy as np
import json
import h5py
import pandas as pd
from scipy import signal

class ViscousPlate(object):

    def __init__(self, config_file, offset): # FIXME: must handle the vomo file

        with open(config_file) as json_data_file:
            conf = json.load(json_data_file)

        print(conf)
        self.offset = offset
        self.conf_pl = conf['plate']
        self.vomo_file = self.conf_pl['vomo_file']

        # compute the non-changing terms in here and save the config parameters
        self.length = 3*self.conf_pl["dim"][1] # increase the period for smoothness
        self.fo = 2*np.pi/self.length

        # compute viscous rigidity and characteristic length
        self.char_t = self.conf_pl["char_time"]*3600*24*365
        self.rigidity = (self.conf_pl["plate_viscosity"]*self.conf_pl["thickness"]**3)/3
        self.alpha = 2*np.pi*self.conf_pl["thickness"]*(self.conf_pl["plate_viscosity"]/(3*self.conf_pl["mantle_viscosity"]))**(1/3)

        # compute spatial values
        self.num_x_elem = int(self.conf_pl["dim"][0]/self.conf_pl['delta']) + 1
        self.num_y_elem = int(self.conf_pl['dim'][1]/self.conf_pl['delta']) + 1

        # must be separated during computation of Green's function
        self.x = np.arange(0, self.conf_pl["dim"][0] + self.conf_pl["delta"], self.conf_pl["delta"])
        self.y = np.arange(0, self.conf_pl['dim'][1] + self.conf_pl["delta"], self.conf_pl["delta"])
        self.y = self.y - self.y.max()/2

        self.vo_gf = np.zeros([self.num_x_elem, self.num_y_elem])
        self.mo_gf = np.zeros_like(self.vo_gf)

        self.theo_deflection = np.zeros([self.num_x_elem, self.num_y_elem])

        #self.vo_debug1 = np.zeros(self.y.shape)
        #self.mo_debug1 = np.zeros(self.y.shape)
        #self.vo_debug2 = np.zeros(self.y.shape)
        #self.mo_debug2 = np.zeros(self.y.shape)

        return

    def compute_greens_function(self, max_coeff1):

        # define shift in y value
        dummy_y = self.y - self.offset
        self.x_space, self.y_space = np.meshgrid(self.x, dummy_y)
        self.x_space = self.x_space.T
        self.y_space = self.y_space.T

        # define and compute for Vo only
        self.Vo = -0.5
        self.Mo = 0 # default is 0

        # compute for the Green's function here
        self.vo_gf = self._compute_case2(max_coeff1)
        self.vo_gf1 = np.copy(self.vo_gf)
        #self.vo_debug1 = np.copy(self.vo_debug2) # debug
        #self.vo_debug2 = np.zeros_like(self.vo_debug2)
        #self.mo_debug2 = np.zeros_like(self.mo_debug2)

        # define and compute for Mo only
        self.Vo = 0
        self.Mo = -0.5

        # compute for the Green's function here
        self.mo_gf = self._compute_case2(max_coeff1)
        self.mo_gf1 = np.copy(self.mo_gf)
        #self.mo_debug1 = np.copy(self.mo_debug2) # debug
        #self.vo_debug2 = np.zeros_like(self.vo_debug2)
        #self.mo_debug2 = np.zeros_like(self.mo_debug2)

        # define another shift in y value
        dummy_y = self.y + self.offset
        self.x_space, self.y_space = np.meshgrid(self.x, dummy_y)
        self.x_space = self.x_space.T
        self.y_space = self.y_space.T

        # define and compute for Vo only
        self.Vo = 0.5
        self.Mo = 0 # default is 0

        # compute for the Green's function here
        self.vo_gf2 = self._compute_case2(max_coeff1)
        self.vo_gf += self.vo_gf2
        #another_dummy = np.copy(self.vo_debug2) # debug
        #self.vo_debug2 = np.zeros_like(self.vo_debug2)
        #self.mo_debug2 = np.zeros_like(self.mo_debug2)

        # define and compute for Mo only
        self.Vo = 0
        self.Mo = 0.5

        # compute for the Green's function here
        self.mo_gf2 = self._compute_case2(max_coeff1)
        self.mo_gf += self.mo_gf2
        #self.vo_debug2 = np.copy(another_dummy) # debug

        return

    def convolve_load(self):

        self.vomo = pd.read_csv(self.vomo_file)

        self.signal_vo = np.zeros_like(self.theo_deflection)
        self.signal_mo = np.zeros_like(self.theo_deflection)

        self.signal_vo[0, :] = self.vomo.iloc[:, 1]
        self.signal_mo[0, :] = self.vomo.iloc[:, 2]

        self.vo_conv = signal.convolve2d(self.signal_vo, self.vo_gf, mode = 'full')
        self.mo_conv = signal.convolve2d(self.signal_mo, self.mo_gf, mode = 'full')

        self.theo_deflection = self.vo_conv + self.mo_conv
        # valid deflection is only a subset of the result
        col_mid = self.theo_deflection.shape[1]//2
        col_lim = self.vo_gf.shape[1]//2
        row_max = self.vo_gf.shape[0]
        self.theo_deflection = self.theo_deflection[0:row_max, col_mid - col_lim: col_mid + col_lim + 1]

    def write_out(self, out_fname, mode = 0):

        """
        mode:
            0 - only total deflection and some attributes
            1 - only  Vo greens function
            2 - only Mo greens function
        """

        out_file = h5py.File(out_fname, 'w')
        if mode == 0: # only total deflection
            z = out_file.create_dataset("deflection", data = np.flipud(self.theo_deflection))
            z = out_file.create_dataset("edge_moment", data = self.vomo.iloc[:, 2])
            out_file.attrs["delta"] = self.conf_pl["delta"]
            out_file.attrs["plate_viscosity"] = self.conf_pl["plate_viscosity"]
            out_file.attrs["mantle_viscosity"] = self.conf_pl["mantle_viscosity"]
            out_file.attrs["rigidity"] = self.rigidity
            out_file.attrs["stress_const"] = 1
            out_file.attrs["poisson"] = self.conf_pl["poisson"]
        elif mode == 1:
            z = out_file.create_dataset("vo gf", data = self.vo_gf)
        elif mode == 2:
            z = out_file.create_dataset("mo gf", data = self.mo_gf)

        out_file.close()

        print("Values in: ", out_fname)

        return

    def _compute_params(self, n, approx = False):

        # sadly, using approximations does not produce good results for X(x)

        new_beta = n*self.fo

        dummy = 1/((new_beta*self.alpha)**4)
        self.kappa = np.sqrt(0.25 + dummy)
        self.gamma1 = np.sqrt(self.kappa + 0.5)
        self.gamma2 = np.sqrt(self.kappa - 0.5)

        self.theta1 = self.kappa**2 + self.kappa*(1-self.conf_pl["poisson"]) - 0.25*self.conf_pl["poisson"]**2
        self.theta2 = self.kappa**2 + 0.5*self.kappa*(1-self.conf_pl["poisson"]) - 0.25*self.conf_pl["poisson"]

        return

    def _compute_AB(self, beta, Vo, Mo):

        self.A = -(Vo*self.gamma1)/(2*self.rigidity*self.theta1*(beta**3))
        self.A +=  Mo*(2*self.kappa + self.conf_pl["poisson"])/(4*self.rigidity*self.theta1*(beta**2))
        self.B = (Vo*(self.conf_pl["poisson"] - 1))/(4*self.rigidity*self.gamma2*self.theta1*(beta**3))
        self.B -= (Mo*self.theta2)/(2*self.gamma1*self.gamma2*self.rigidity*self.theta1*(beta**2))

        return

    def _compute_case2(self, max_coeff):

        # get coeffs
        coeffs = np.arange(1, max_coeff + 1)
        div = coeffs%2
        coeffs = coeffs[div != 0]

        partial_sol = np.zeros(self.x_space.shape)

        for x in coeffs:
            self._compute_params(x)
            Vo = 4*self.Vo/(np.pi*x)
            Mo = 4*self.Mo/(np.pi*x)
            beta = x*self.fo

            #self.vo_debug2 += Vo*np.sin(beta*self.y_space[-1, :])
            #self.mo_debug2 += Mo*np.sin(beta*self.y_space[-1, :])


            self._compute_AB(beta, Vo, Mo)
            x_sol = np.exp(-beta*self.gamma1*self.x_space)*(self.A*np.cos(beta*self.gamma2*self.x_space) + self.B*np.sin(beta*self.gamma2*self.x_space))
            y_sol = np.sin(beta*self.y_space)

            partial_sol += x_sol*y_sol

        return partial_sol

    def compute_numerical_secder(self):

        self.num_der_xx = np.zeros_like(self.theo_deflection)
        self.num_der_yy = np.zeros_like(self.theo_deflection)
        self.num_der_xy = np.zeros_like(self.theo_deflection)

        for i in range(1, self.num_der_xx.shape[0] - 1):
            for j in range(1, self.num_der_xx.shape[1] - 1):
                self.num_der_xx[i, j] = (self.theo_deflection[i-1,j] - 2*self.theo_deflection[i, j] +
                self.theo_deflection[i+1, j])/(self.conf_pl['delta']**2)
                self.num_der_yy[i, j] = (self.theo_deflection[i,j-1] - 2*self.theo_deflection[i, j] +
                self.theo_deflection[i, j+1])/(self.conf_pl['delta']**2)
                self.num_der_xy[i, j] = (self.theo_deflection[i-1,j-1] - self.theo_deflection[i-1, j+1] -
                self.theo_deflection[i+1, j-1] + self.theo_deflection[i+1, j+1])/(self.conf_pl['delta']**2)

        self.x_stress = -(12*self.rigidity/self.conf_pl["thickness"]**3)*(self.num_der_xx + self.conf_pl["poisson"]*self.num_der_yy)
        self.y_stress = -(12*self.rigidity/self.conf_pl["thickness"]**3)*(self.num_der_yy + self.conf_pl["poisson"]*self.num_der_xx)
        self.xy_stress = -(12*self.rigidity/self.conf_pl["thickness"]**3)*((1-self.conf_pl["poisson"])*self.num_der_xy)

        return


