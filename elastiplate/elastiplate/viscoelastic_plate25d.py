import numpy as np
import pandas as pd
import json, h5py

from scipy.linalg import det, inv, svd
from scipy.sparse import csr_matrix, coo_matrix, diags, save_npz, load_npz
from scipy.sparse.linalg import spsolve, lsqr
from scipy import interpolate

# WTF, this is the fourth plate model - viscoelastic plate.
# Hopefully this is also the last
# This one also uses scaling

class ViscoelasticPlate25D(object):


    def __init__(self, config_file, diff_visc = True, debug = False, gen_a_matrix = False):

        # unpack parameters in here
        with open(config_file) as json_data_file:
            conf = json.load(json_data_file)

        self.gen_a_matrix = gen_a_matrix
        self.debug = debug
        print(conf)
        conf_pl = conf['plate']

        # read vo and mo
        self.vomo_file = conf_pl["vomo_file"]
        self.vomo = pd.read_csv(self.vomo_file)

        self.num_step = conf_pl["num_step"]
        #self.t_c = 3*conf_pl["plate_viscosity"]/conf_pl["modulus"]
        self.t_c = 4*(1-conf_pl["poisson"]**2)*conf_pl["plate_viscosity"]/conf_pl["modulus"]

        # elastic parameters
        self.rigidity = (conf_pl["modulus"]*conf_pl["thickness"]**3)/(12*(1-conf_pl["poisson"]**2))
        self.alpha = (4*self.rigidity/((conf_pl["rho_mantle"] - conf_pl["rho_infill"])*conf_pl['gravity']))**0.25
        self.visc_rigidity = (conf_pl["plate_viscosity"]*conf_pl["thickness"]**3)/3

        if diff_visc:
            self.visc_rigidity = (conf_pl["plate_viscosity"]*conf_pl["thickness"]**3)/3
            self.beta = (conf_pl["plate_viscosity"]**4/(6*conf_pl["mantle_viscosity"]**4))**(1/3)
            self.beta = conf_pl["thickness"]*((self.beta/3)**0.25)

            self.ratio_param = (self.beta/self.alpha)**4
        else:
            self.ratio_param = 1

        # parameters needed for normalization
        if diff_visc:
            self.x_c = self.beta/np.sqrt(2)
        else:
            self.x_c = self.alpha/np.sqrt(2)

        self.y_c = self.x_c
        self.w_c = self.x_c

        self.chi_1 = self.vomo.iloc[:, 2]*self.x_c*self.t_c/self.visc_rigidity # bending moment bali ang indexing!!!! -_-
        self.chi_2 = self.vomo.iloc[:, 1]*self.t_c*(self.x_c**2)/self.visc_rigidity # shear

        # old version
        #self.chi_1 = self.vomo.iloc[:, 2]*self.x_c/self.rigidity # bending moment bali ang indexing!!!! -_-
        #self.chi_2 = self.vomo.iloc[:, 1]*(self.x_c**2)/self.rigidity # shear

        self.num_x_elem = int(conf_pl["dim"][0]/conf_pl['delta']) + 1
        self.num_y_elem = int(conf_pl['dim'][1]/conf_pl['delta']) + 1
        self.x = np.arange(0, conf_pl["dim"][0] + conf_pl["delta"], conf_pl["delta"])/self.x_c
        self.y = np.arange(0, conf_pl['dim'][1] + conf_pl["delta"], conf_pl["delta"])/self.y_c
        self.a_matrix = np.zeros([self.num_x_elem, self.num_x_elem])

        self.og_x_delta = conf_pl["delta"]
        self.delta = conf_pl["delta"]/self.x_c

        self.t_delta = conf_pl["t_delta"]*1e3 # config file in terms of ky
        self.time_ky_val = np.arange(0, self.num_step*self.t_delta, self.t_delta)
        self.t_delta = self.t_delta*31556926

        # compute the moment and shear rate applied on the edge of the plate
        self.Mo_rate = self.vomo.iloc[:, 2]/self.t_delta
        self.Vo_rate = self.vomo.iloc[:, 1]/self.t_delta

        self.xi_1 = self.Mo_rate*(self.t_c**2)*self.x_c/self.visc_rigidity
        self.xi_2 = self.Vo_rate*(self.t_c**2)*(self.x_c**2)/self.visc_rigidity

        # original
        #self.xi_1 = self.Mo_rate*self.t_c*self.x_c/self.rigidity
        #self.xi_2 = self.Vo_rate*self.t_c*(self.x_c**2)/self.rigidity

        self.og_t_delta = self.t_delta
        self.t_delta = self.t_delta/self.t_c
        self.poisson = conf_pl["poisson"]

        return

    def rescale_out(self):

        self.x_resc = self.x_c*self.x
        self.y_resc = self.y_c*self.y
        self.def_rec_resc = self.w_c*self.def_rec_norm

        return

    def _gen_loading(self, chi_1, chi_2, xi_1 = 0, xi_2 = 0, deflection = None): # deflection is not none if using viscous beam
        # xi_1 and xi_2 are only needed for the first iteration (technically 2nd)

        self.load_array = np.zeros(self.num_x_elem)

        term1 = chi_1*self.t_delta*self.delta**2 + xi_1*self.t_delta*self.delta**2
        term2 = chi_2*self.t_delta*self.delta**3 + xi_2*self.t_delta*self.delta**3
        deltax4 = self.delta**4

        for i in range(self.num_x_elem):
            if i == 0:
                self.load_array[i] = np.sum(np.array([-8, 1])*deflection[1:3])
            elif i == 1:
                self.load_array[i] = np.sum(np.array([7 + deltax4, -4, 1])*deflection[1:4])
            elif i == self.num_x_elem - 2:
                self.load_array[i] = term1 + np.sum(np.array([1, -4, 5 + deltax4, -2])*deflection[i-2:i+2])
            elif i == self.num_x_elem - 1:
                self.load_array[i] = -2*term1 - 2*term2 + np.sum(np.array([2, -4, 2 + deltax4])*deflection[i-2:i+1])
            else:
                self.load_array[i] = np.sum(np.array([1, -4, 6 + deltax4, -4, 1])*deflection[i-2:i+3])

        return

    def fill_matrix(self):

        d4 = (1 + self.ratio_param*self.t_delta)*self.delta**4

        for i in range(self.num_x_elem):
            if i == 0:
                self.a_matrix[i, i+1:i+3] = np.array([-8, 1])
            elif i == 1:
                self.a_matrix[i, i:i+3] = np.array([7+d4, -4, 1])
                #self.a_matrix[i, i-1:i+3] = np.array([-4, 7+d4, -4, 1])
            elif i == self.num_x_elem - 2:
                self.a_matrix[i, i-2:i+2] = np.array([1, -4, 5+d4, -2])
            elif i == self.num_x_elem - 1:
                self.a_matrix[i, i-2:i+1] = np.array([2, -4, 2+d4])
            else:
                self.a_matrix[i, i-2:i+3] = np.array([1, -4, 6+d4, -4, 1])

        #self.a_matrix = np.flipud(self.a_matrix)
        return


    def solve_viscoelastic(self):

        self.fill_matrix()

        self.def_rec_norm = np.zeros([self.num_step, self.num_x_elem, self.num_y_elem])

        for i in np.arange(self.num_step):
            print("iteration: ", i)
            if i == 0:
                None # do nothing
            else:
                for j in np.arange(self.num_y_elem):
                    #if i == 0:
                    #     self._gen_loading(  chi_1 = 0,  # original
                    #                         chi_2 = 0, 
                    #                         xi_1 = self.xi_1[j], 
                    #                         xi_2 = self.xi_2[j], 
                    #                         deflection = self.def_rec_norm[i-1, :, j])
                    #else:
                    if i < 20:
                        self._gen_loading(  chi_1 = self.chi_1[j],  # original
                                            chi_2 = self.chi_2[j], 
                                            xi_1 = 0, 
                                            xi_2 = 0, 
                                            deflection = self.def_rec_norm[i-1, :, j])
                    # elif i == 20:
                    #     self._gen_loading(  chi_1 = self.chi_1[j],  # original
                    #                         chi_2 = self.chi_2[j], 
                    #                         xi_1 = -self.xi_1[j], 
                    #                         xi_2 = -self.xi_2[j], 
                    #                         deflection = self.def_rec_norm[i-1, :, j])
                    else:
                        self._gen_loading(  chi_1 = 0, 
                                            chi_2 = 0, 
                                            xi_1 = 0, 
                                            xi_2 = 0, 
                                            deflection = self.def_rec_norm[i-1, :, j])

                    # original 
                    #if i == 1:
                    #    self._gen_loading(self.chi_1[j], self.chi_2[j], self.xi_1[j], self.xi_2[j], deflection = self.def_rec_norm[i-1, :, j])
                    #else:
                    #    self._gen_loading(self.chi_1[j], self.chi_2[j], 0, 0, deflection = self.def_rec_norm[i-1, :, j]) # first argument is no longer needed
                    self.solve_spsolve()
                    self.def_rec_norm[i, :, j] = self.deflection

        return

    def solve_spsolve(self):

        csr_mat = csr_matrix(self.a_matrix)
        self.deflection = spsolve(csr_mat, self.load_array)
        return

    def write_out(self, out_fname):

        """
        Writes the deflection and some data attributes to out_fname.
        """

        out_file = h5py.File(out_fname, 'w')
        z = out_file.create_dataset("deflection", data = self.def_rec_resc)

        z = out_file.create_dataset("Mo", data = self.vomo.iloc[:, 2])
        z = out_file.create_dataset("Vo", data = self.vomo.iloc[:, 1])
        z = out_file.create_dataset("Mo_rate", data = self.Mo_rate)
        z = out_file.create_dataset("Vo_rate", data = self.Vo_rate)
        z = out_file.create_dataset("x", data = self.x_resc)
        z = out_file.create_dataset("y", data = self.y_resc)
        z = out_file.create_dataset("time_ky", data = self.time_ky_val)

        out_file.attrs["xy_delta"] = self.og_x_delta
        out_file.attrs["t_delta"] = self.og_t_delta
        out_file.attrs["rigidity"] = self.rigidity
        out_file.attrs["poisson"] = self.poisson
        out_file.attrs["t_maxwell"] = self.t_c

        out_file.close()

        print("Deflection values in: ", out_fname)

        return

    # this thing works but should not be used
    def solve_25D(self):

        self.deflection_25D = np.zeros([self.num_x_elem, self.num_y_elem])

        x = np.flip(self.x_resc)

        for i in np.arange(0, self.num_y_elem):
            Vo = self.vomo.iloc[i, 1]
            Mo = self.vomo.iloc[i, 2]
            w_c = -(self.alpha**2/(2*self.rigidity))*(Vo*self.alpha + Mo)
            if (Vo == 0) and (Mo == 0):
                self.deflection_25D[:, i] = 0
            else:
                chi = -Mo/(Vo*self.alpha + Mo)
                self.deflection_25D[:, i] =w_c*np.exp(-x/np.sqrt(2))*(chi*np.sin(x/np.sqrt(2)) + np.cos(x/np.sqrt(2)))


            #self.deflection_25D[:, i] = np.exp(-x/np.sqrt(2))*(-self.edge_moment[i]*np.sin(x/np.sqrt(2)) + np.cos(x/np.sqrt(2)))

        return

