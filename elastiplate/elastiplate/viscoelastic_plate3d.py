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

class ViscoelasticPlate3D(object):


    def __init__(self, config_file, diff_visc = True, debug = False, gen_a_matrix = False):
        """
        diff_visc - if set to True, we use a different value for the velocity-dependent resistance
        """

        # unpack parameters in here
        with open(config_file) as json_data_file:
            conf = json.load(json_data_file)

        print(conf)
        conf_pl = conf['plate']
        self.debug = debug
        self.gen_a_matrix = gen_a_matrix
        self.a_matrix_file = conf_pl['a_mat_file']

        # read vo and mo
        self.vomo_file = conf_pl["vomo_file"]
        self.vomo = pd.read_csv(self.vomo_file)

        self.num_step = conf_pl["num_step"]
        #self.t_c = 3*conf_pl["plate_viscosity"]/conf_pl["modulus"]
        self.t_c = 4*(1-conf_pl['poisson']**2)*conf_pl['plate_viscosity']/conf_pl['modulus']

        # elastic parameters
        self.rigidity = (conf_pl["modulus"]*conf_pl["thickness"]**3)/(12*(1-conf_pl["poisson"]**2))
        self.alpha = (4*self.rigidity/((conf_pl["rho_mantle"] - conf_pl["rho_infill"])*conf_pl['gravity']))**0.25
        self.visc_rigidity = (conf_pl["plate_viscosity"]*conf_pl["thickness"]**3)/3

        # viscous parameters
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

        self.chi_1 = self.vomo.iloc[:, 2]*self.t_c*self.x_c/self.visc_rigidity # bending moment
        self.chi_2 = self.vomo.iloc[:, 1]*self.t_c*(self.x_c**2)/self.visc_rigidity # shear

        self.num_x_elem = int(conf_pl["dim"][0]/conf_pl['delta']) + 1
        self.num_y_elem = int(conf_pl['dim'][1]/conf_pl['delta']) + 1
        self.x = np.arange(0, conf_pl["dim"][0] + conf_pl["delta"], conf_pl["delta"])/self.x_c
        self.y = np.arange(0, conf_pl['dim'][1] + conf_pl["delta"], conf_pl["delta"])/self.y_c
        self.a_matrix = np.zeros([self.num_x_elem*self.num_y_elem, self.num_x_elem*self.num_y_elem])

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

        self.og_t_delta = self.t_delta
        self.t_delta = self.t_delta/self.t_c

        # what to do here?

        #self._gen_loading(conf_pl) # might no longer be needed for viscoelastic plate
        #self._solve_analytical(conf_pl, conf_ld, conf_mo) # might no longer be needed for viscoelastic plate

        # initialize plate matrix
        #self.deflection_25D = np.zeros([self.num_x_elem, self.num_y_elem]) # might no longer be needed for viscoelastic plate

        # D.E. stencils used
        self.poisson = conf_pl['poisson']
        p = conf_pl['poisson']
        p2 = p**2
        self.bc1 = np.array([[0., 0, 1, 0, 0], [0, 0, -8, 2, 0], [0, 0, 21, -8, 1], [0, 0, -8, 2, 0], [0, 0, 1, 0, 0]])
        self.bc2 = np.array([[0., 0, 1, 0, 0], [0, 0, -8, 2, 0], [0, 0, 20, -8, 1], [0, 0, -6 + 2*p, 2 - p, 0], [0, 0, 0, 0, 0]])
        self.bc3 = np.array([[0., 0, 2, 0, 0], [0, 0, -12 + 4*p, 4 - 2*p, 0], [0, 0, 17 - 8*p - 5*p2, -8 + 4*p + 4*p2, 1 - p2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.de_stencil = np.array([[0., 0, 1, 0, 0], [0, 2, -8, 2, 0], [1, -8, 20, -8, 1], [0, 2, -8, 2, 0], [0, 0, 1, 0, 0]])
        self.s2 = np.array([[0., 0, 2, 0, 0], [0, 4 - 2*p, -12 + 4*p, 4 - 2*p, 0], [1 - p2, -8 + 4*p + 4*p2, 16 - 8*p -6*p2, -8 + 4*p + 4*p2, 1- p2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.bef_s2 = np.array([[0., 0, 1, 0, 0], [0, 2, -8, 2, 0], [1, -8, 19, -8, 1], [0, 2 - p, -6 + 2*p, 2 - p, 0], [0, 0, 0, 0, 0]])
        self.rb_corner = np.array([[0., 0, 2 - 2*p2, 0, 0], [0, 8 - 8*p, -12 + 8*p + 4*p2, 0, 0], [2 - 2*p2, -12 + 8*p + 4*p2, 12 - 8*p - 4*p2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.bef_rb_corner = np.array([[0., 0, 1, 0, 0], [0, 2, -8, 2 - p, 0], [1, -8, 18, -6 + 2*p, 0], [0, 2 - p, -6 + 2*p, 2 - 2*p, 0], [0, 0, 0, 0, 0]])
        self.top_rb_corner = np.array([[0., 0, 1-p2, 0, 0], [0, 4-2*p, -8+4*p+4*p2, 0, 0], [2, -12+4*p, 15-8*p - 5*p2, 0, 0], [0, 4 - 2*p, -6 + 4*p + 2*p2, 0, 0], [0, 0, 0, 0, 0]])

        return


    def rescale_out(self):

        self.x_resc = self.x_c*self.x
        self.y_resc = self.y_c*self.y
        self.def_rec_resc = self.w_c*self.def_rec_norm

        return

    def _gen_loading(self, step_num, deflection = None):

        self.load = np.zeros([self.num_x_elem, self.num_y_elem])
        self.edge_moment = np.zeros(self.num_y_elem)
        self.tot_moment = np.zeros([2, self.num_y_elem])

        # this part is for the contributions of the shear and moment distributions
        # only supports load and moment specified at each point
        # transverse load in here - shear in here - this is term 2
        if step_num == 0:
            print("step_num = 0 is not handled by this method!!")
            return
        elif step_num < 20: # apply the positive rate
            self.load[-1, :] = self.t_delta*(self.chi_2) # shear  - original code
            self.edge_moment[:] = self.t_delta*(self.chi_1) # 
        
        # elif step_num == 2: # constant value
        #     # self.load[-1, :] = self.t_delta*(self.chi_2 + self.xi_2) # shear  - original code
        #     # self.edge_moment[:] = self.t_delta*(self.chi_1 + self.xi_1) # 
        
        
        #     self.load[-1, :] = self.t_delta*(self.chi_2) # use this
        #     self.edge_moment[:] = self.t_delta*(self.chi_1) #
        # elif step_num == 3: # negative rate
        #     self.load[-1, :] = self.t_delta*(-self.xi_2) 
        #     self.edge_moment[:] = self.t_delta*(-self.xi_1) 
        else:
            # 21/01/2022 - if not the first time step, applied shearing force and moment are zero
            self.load[-1, :] = 0
            self.edge_moment[:] = 0
            # end edit

            # original 
            # self.load[-1, :] = self.t_delta*self.chi_2
            # self.edge_moment[:] = self.t_delta*self.chi_1

        self.edge_field = -2*self.load*self.delta**3 # 14/03/2020 - changed to negative

        self.tot_moment[0, :] = self.edge_moment*self.delta**2
        self.tot_moment[1, :] = (-2-2*self.poisson)*self.edge_moment*self.delta**2
        self.tot_moment[1, 2:-2] += self.poisson*(self.edge_moment[3:-1]*self.delta**2) # east
        self.tot_moment[1, 2:-2] += self.poisson*(self.edge_moment[1:-3]*self.delta**2) # west

        # edges
        self.tot_moment[1, 0] += 2*self.poisson*self.edge_moment[1]*self.delta**2
        self.tot_moment[1, -1] += 2*self.poisson*self.edge_moment[-2]*self.delta**2

        # penultimate
        self.tot_moment[1, 1] += self.poisson*self.edge_moment[2]*self.delta**2
        self.tot_moment[1, -2] += self.poisson*self.edge_moment[-3]*self.delta**2

        self.edge_field[-2:,:] += self.tot_moment # this is used for solving the DE

        print('range of edge_field before: ', self.edge_field.min(), self.edge_field.max())

        # this part is for getting the contribution of the previous time step
        if step_num < 2: # no need to add if this is the first - deflection arg is don't care
            None
        else: # need to add contribution from previous deflection - first is False
            for i in range(self.num_x_elem):
                for j in range(self.num_y_elem):

                    cp_stencil = self._eval_bndry(i, j, add_t_delta = False)

                    row = np.repeat(np.arange(i - 2, i + 3, 1), 5).reshape(5, 5)

                    col = np.array([[x for x in range(j - 2, j + 3)]])
                    col = np.repeat(col, 5, axis = 0).reshape(5, 5)

                    k, l = np.where(np.abs(cp_stencil) != 0) # remember this method! It is useful!!!

                    defl_nonzero = deflection[row[k, l], col[k, l]]
                    sten_nonzero = cp_stencil[k, l]

                    if self.debug:
                        if i > 0.8*self.num_x_elem:
                            print('i: ', i, ' j: ', j)
                            print(cp_stencil)
                            print(row)
                            print(col)
                            print(k)
                            print(l)
                            print('defl_nonzero: ', defl_nonzero)
                            print('sten_nonzero: ', sten_nonzero)
                            print('sum: ', np.sum(defl_nonzero*sten_nonzero))

                    self.edge_field[i, j] += np.sum(defl_nonzero*sten_nonzero) # if empty, sum is zero

        print('range of edge_field: ', self.edge_field.min(), self.edge_field.max())
        return

    # easier if you just hardcode stuff - although longer - this is the cantilever plate case
    def _eval_bndry(self, x, y, add_t_delta = False): # loc_row and loc_col are 2D arrays

        if x == 0:
            if y == 0: # side 1
                cp_stencil = np.copy(self.bc3)
                cp_stencil = np.fliplr(cp_stencil.T)
            elif y == 1: # side 1
                cp_stencil = np.copy(self.bc2)
                cp_stencil = np.fliplr(cp_stencil.T)

            elif y == self.num_y_elem - 1: # side 3
                cp_stencil = np.copy(self.bc3)
                cp_stencil = cp_stencil.T

            elif y == self.num_y_elem - 2: # side 3
                cp_stencil = np.copy(self.bc2)
                cp_stencil = cp_stencil.T
            else:
                cp_stencil = np.copy(self.bc1)
                cp_stencil = cp_stencil.T


        elif x == 1: #
            if y == 0: # side 1
                cp_stencil = np.copy(self.s2)
                cp_stencil = np.fliplr(np.flipud(np.rot90(cp_stencil)))

            elif y == 1: # side 1
                cp_stencil = np.copy(self.bef_s2)
                cp_stencil = np.fliplr(np.flipud(np.rot90(cp_stencil)))

            elif y == self.num_y_elem - 1: # side 3
                cp_stencil = np.copy(self.s2)
                cp_stencil = np.rot90(cp_stencil)

            elif y == self.num_y_elem - 2: # side 3
                cp_stencil = np.copy(self.bef_s2)
                cp_stencil = np.rot90(cp_stencil)
            else:
                cp_stencil = np.copy(self.de_stencil)


            cp_stencil[0, :] = 0
            cp_stencil[2, 2] += 1

        elif x == self.num_x_elem - 1:
            if y == 0:
                cp_stencil = np.copy(self.rb_corner)
                cp_stencil = np.fliplr(cp_stencil)
            elif y == 1:
                cp_stencil = np.copy(self.top_rb_corner)
                cp_stencil = np.fliplr(cp_stencil.T)
            elif y == self.num_y_elem - 1:
                cp_stencil = np.copy(self.rb_corner)
            elif y == self.num_y_elem - 2:
                cp_stencil = np.copy(self.top_rb_corner)
                cp_stencil = cp_stencil.T
            else:
                cp_stencil = np.copy(self.s2)

        elif x == self.num_x_elem - 2:
            if y == 0:
                cp_stencil = np.copy(self.top_rb_corner)
                cp_stencil = np.fliplr(cp_stencil)
            elif y == 1:
                cp_stencil = np.copy(self.bef_rb_corner)
                cp_stencil = np.fliplr(cp_stencil)
            elif y == self.num_y_elem - 1:
                cp_stencil = np.copy(self.top_rb_corner)
            elif y == self.num_y_elem - 2:
                cp_stencil = np.copy(self.bef_rb_corner)
            else:
                cp_stencil = np.copy(self.bef_s2)
        else:
            if y == 0: # side 1
                cp_stencil = np.copy(self.s2)
                cp_stencil = np.fliplr(np.flipud(np.rot90(cp_stencil)))
            elif y == 1: # side 1
                cp_stencil = np.copy(self.bef_s2)
                cp_stencil = np.fliplr(np.flipud(np.rot90(cp_stencil)))
            elif y == self.num_y_elem - 1: # side 3
                cp_stencil = np.copy(self.s2)
                cp_stencil = np.rot90(cp_stencil)
            elif y == self.num_y_elem - 2: # side 3
                cp_stencil = np.copy(self.bef_s2)
                cp_stencil = np.rot90(cp_stencil)
            else:
                cp_stencil = np.copy(self.de_stencil)

        if add_t_delta:
            None
            # old code
            #cp_stencil[2, 2] += (1 + self.ratio_param*self.t_delta)*self.delta**4
        else:
            cp_stencil[2, 2] += self.delta**4

        return cp_stencil

    def fill_matrix(self):

        if self.gen_a_matrix:
            for i in range(self.num_x_elem):
                for j in range(self.num_y_elem):

                    arr_idx = i*self.num_y_elem + j

                    cp_stencil = self._eval_bndry(i, j, add_t_delta = True)
                    #add_t_delta = True does not add anything

                    row = np.repeat(np.arange(i - 2, i + 3, 1), 5).reshape(5, 5)

                    col = np.array([[x for x in range(j - 2, j + 3)]])
                    col = np.repeat(col, 5, axis = 0).reshape(5, 5)

                    idx_1d = row*self.num_y_elem + col
                    idx_1d[cp_stencil == 0] = -1

                    idx_1d = idx_1d[idx_1d > -1]
                    coeffs = cp_stencil[cp_stencil != 0]
                    arr_idx = np.repeat(arr_idx, idx_1d.shape[0])
                    if i == 0 and j == 0:
                        self.a_mat_row = arr_idx
                        self.a_mat_col = idx_1d
                        self.a_mat_val = coeffs
                    else:
                        self.a_mat_row = np.append(self.a_mat_row, arr_idx)
                        self.a_mat_col = np.append(self.a_mat_col, idx_1d)
                        self.a_mat_val = np.append(self.a_mat_val, coeffs)

            self.a_matrix = coo_matrix((self.a_mat_val, (self.a_mat_row, self.a_mat_col)), shape = (self.num_x_elem*self.num_y_elem, self.num_x_elem*self.num_y_elem))
            save_npz(self.a_matrix_file, self.a_matrix)
        else: # read from a_mat_file
            self.a_matrix = load_npz(self.a_matrix_file)

        # add the iso term to the diagonal
        self.iso_term = (1 + self.ratio_param*self.t_delta)*self.delta**4
        self.diag_mat = np.repeat(self.iso_term, self.num_x_elem*self.num_y_elem)
        self.diag_mat = diags(self.diag_mat, 0)

        self.a_matrix = self.a_matrix + self.diag_mat

        return

    def solve_viscoelastic(self):

        self.fill_matrix()

        self.def_rec_norm = np.zeros([self.num_step, self.num_x_elem, self.num_y_elem])

        for i in np.arange(self.num_step):
            print("iteration: ", i)
            if i == 0:
                None # do nothing
            else:
                self._gen_loading(step_num = i, deflection = self.def_rec_norm[i-1, :, :])
                self.solve_spsolve()
                self.def_rec_norm[i, :, :] = self.deflection

        return

    def solve_spsolve(self):

        # faster way to solve
        #inv_mat = inv(self.a_matrix)

        csr_mat = csr_matrix(self.a_matrix)
        self.deflection = spsolve(csr_mat, self.edge_field.reshape(self.num_x_elem*self.num_y_elem).T)
        self.deflection = self.deflection.reshape([self.num_x_elem, self.num_y_elem])   # should not be negative as this will give the wrong stresses

        return

    def solve_lsqr(self):

        test_out = lsqr(self.a_matrix, self.edge_field.reshape(self.num_x_elem*self.num_y_elem))
        self.deflection = test_out[0].reshape(self.num_x_elem, self.num_y_elem)

        return

    def solve_svd(self):

        U,s,Vh = svd(self.a_matrix)
        pinv_svd = np.dot(np.dot(Vh.T,inv(np.diag(s))),U.T)
        self.deflection = np.dot(pinv_svd, self.edge_field.reshape(self.num_x_elem*self.num_y_elem))
        self.deflection = self.deflection.reshape(self.num_x_elem, self.num_y_elem)

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
        out_file.attrs["visc_rigidity"] = self.visc_rigidity
        out_file.attrs["poisson"] = self.poisson
        out_file.attrs["t_maxwell"] = self.t_c

        out_file.close()

        print("Deflection values in: ", out_fname)

        return

