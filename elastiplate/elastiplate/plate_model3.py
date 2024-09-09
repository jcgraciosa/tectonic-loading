import numpy as np
import pandas as pd
import json, h5py

from scipy.linalg import det, inv, svd
from scipy.sparse import csr_matrix, coo_matrix, diags, save_npz, load_npz
from scipy.sparse.linalg import spsolve, lsqr
from scipy import interpolate

# scaled plate model

class ScaledPlate(object):


    def __init__(self, config_file, pl_type = 'elastic', gen_a_matrix = False):

        # unpack parameters in here
        with open(config_file) as json_data_file:
            conf = json.load(json_data_file)

        print(conf)
        conf_pl = conf['plate']

        # a_matrix variables
        self.gen_a_matrix = gen_a_matrix
        self.a_matrix_file = conf_pl['a_mat_file']

        # read vo and mo
        self.vomo_file = conf_pl['vomo_file']
        self.vomo = pd.read_csv(self.vomo_file)

        # old code
        #self.vo_amp = self.vomo.iloc[:, 1].min()    # use the minimum value since Vo is expected to be less than 0
        #self.mo_amp = self.vomo.iloc[:, 2].min()    # we use the minimum since Mo is expected to be less than 0
        #if self.mo_amp > 0:
        #    print("Warning: minimum value of Mo is positive...")

        # get back to these

        if pl_type == 'elastic':
            self.rigidity = (conf_pl["modulus"]*conf_pl["thickness"]**3)/(12*(1-conf_pl["poisson"]**2))
            self.alpha = (4*self.rigidity/((conf_pl["rho_mantle"] - conf_pl["rho_infill"])*conf_pl['gravity']))**0.25
        elif pl_type == 'viscous':
            self.rigidity = (conf_pl["plate_viscosity"]*conf_pl["thickness"]**3)/3
            self.alpha = conf_pl["thickness"]*(conf_pl["plate_viscosity"]/(3*conf_pl["mantle_viscosity"]))**(1/3)

        # parameters needed for normalization
        self.x_c = self.alpha/np.sqrt(2)
        self.y_c = self.x_c
        self.w_c = self.x_c

        self.chi_1 = self.vomo.iloc[:, 2]*self.x_c/self.rigidity # bending moment bali ang indexing!!!! -_-
        self.chi_2 = self.vomo.iloc[:, 1]*(self.x_c**2)/self.rigidity # shear

        # old codes - chi_1 - Vo, chi_2 - Mo
        #self.chi_1 = self.vo_amp*self.alpha/(np.sqrt(2)*(self.vo_amp*self.alpha + self.mo_amp))
        #self.chi_2 = self.mo_amp/(self.vo_amp*self.alpha + self.mo_amp)

        self.num_x_elem = int(conf_pl["dim"][0]/conf_pl['delta']) + 1
        self.num_y_elem = int(conf_pl['dim'][1]/conf_pl['delta']) + 1
        print(self.num_x_elem)
        print(self.num_y_elem)

        self.x = np.arange(0, conf_pl["dim"][0] + conf_pl["delta"], conf_pl["delta"])/self.x_c
        self.y = np.arange(0, conf_pl['dim'][1] + conf_pl["delta"], conf_pl["delta"])/self.y_c

        self.delta = conf_pl["delta"]/self.x_c # copied for resolution test
        self.iso_term = self.delta**4
        self.stress_const = 12*self.rigidity/conf_pl["thickness"]**3

        self._gen_loading(conf_pl)
        #self._solve_analytical(conf_pl, conf_ld, conf_mo)

        # initialize plate matrix
        self.deflection_25D = np.zeros([self.num_x_elem, self.num_y_elem])

        # set-up constant matrices here
        self.x_base = np.repeat(np.arange(-2, 3, 1), 5)
        self.y_base = np.array([[x for x in range(-2, 3)]])
        self.y_base = np.repeat(self.y_base, 5, axis = 0).reshape(25)

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

        # moment, shear stress stencils used
        self.p = conf_pl["poisson"]
        self.delta = conf_pl["delta"]

    def rescale_out(self):

        self.x_resc = self.x_c*self.x
        self.y_resc = self.y_c*self.y
        self.deflection_resc = self.w_c*self.deflection

    def _gen_loading(self, conf_pl):

        self.load = np.zeros([self.num_x_elem, self.num_y_elem])
        self.edge_moment = np.zeros(self.num_y_elem)
        self.tot_moment = np.zeros([2, self.num_y_elem])

        # only supports load and moment specified at each point
        # transverse load in here
        #self.load[-1, :] = self.chi_1*(self.vomo.iloc[:, 1]/np.abs(self.vo_amp)) # old code - for reference

        self.load[-1, :] = self.chi_2
        self.edge_moment[:] = self.chi_1

        self.edge_field = -(2*self.load*self.delta**3) # 14/03/2020 - changed to negative

        # edge moment
        # old code - for reference only
        #self.edge_moment[:] = self.chi_2*(self.vomo.iloc[:, 2]/np.abs(self.mo_amp)) # will go back to being negative - careful with the signs!

        self.tot_moment[0, :] = self.edge_moment*self.delta**2
        self.tot_moment[1, :] = (-2-2*conf_pl['poisson'])*self.edge_moment*self.delta**2
        self.tot_moment[1, 2:-2] += conf_pl['poisson']*(self.edge_moment[3:-1]*self.delta**2) # east
        self.tot_moment[1, 2:-2] += conf_pl['poisson']*(self.edge_moment[1:-3]*self.delta**2) # west

        # edges
        self.tot_moment[1, 0] += 2*conf_pl['poisson']*self.edge_moment[1]*self.delta**2
        self.tot_moment[1, -1] += 2*conf_pl['poisson']*self.edge_moment[-2]*self.delta**2

        # penultimate
        self.tot_moment[1, 1] += conf_pl['poisson']*self.edge_moment[2]*self.delta**2
        self.tot_moment[1, -2] += conf_pl['poisson']*self.edge_moment[-3]*self.delta**2

        self.edge_field[-2:,:] += self.tot_moment # this is used for solving the DE

        return

    # easier if you just hardcode stuff - although longer - this is the cantilever plate case
    def _eval_bndry(self, x, y): # loc_row and loc_col are 2D arrays

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

        #cp_stencil[2, 2] += self.iso_term
        #print(x, y)
        #print(cp_stencil)
        return cp_stencil

    def fill_matrix(self):

        if self.gen_a_matrix:
            for i in range(self.num_x_elem):
                for j in range(self.num_y_elem):

                    arr_idx = i*self.num_y_elem + j

                    cp_stencil = self._eval_bndry(i, j)

                    row = np.repeat(np.arange(i - 2, i + 3, 1), 5).reshape(5, 5)

                    col = np.array([[x for x in range(j - 2, j + 3)]])
                    col = np.repeat(col, 5, axis = 0).reshape(5, 5)


                    idx_1d = row*self.num_y_elem + col
                    idx_1d[cp_stencil == 0] = -1

                    idx_1d = idx_1d[idx_1d > -1]
                    #print(idx_1d)
                    coeffs = cp_stencil[cp_stencil != 0]
                    #print(coeffs)
                    arr_idx = np.repeat(arr_idx, idx_1d.shape[0])
                    if i == 0 and j == 0: # initialization
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
        self.diag_mat = np.repeat(self.iso_term, self.num_x_elem*self.num_y_elem)
        self.diag_mat = diags(self.diag_mat, 0)

        self.a_matrix = self.a_matrix + self.diag_mat

        return

    def solve_spsolve(self):

        # faster way to solve
        #inv_mat = inv(self.a_matrix)

        csr_mat = csr_matrix(self.a_matrix)
        #csr_mat = self.a_matrix
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

    def write_out(self, out_fname, mode = 0):

        """If mode = 0, writing the deflection of the 3D model.
        If mode = 1, writing the deflection of the 2.5D model."""

        out_file = h5py.File(out_fname, 'w')
        if mode == 0:
            z = out_file.create_dataset("deflection", data = self.deflection_resc)
        else:
            z = out_file.create_dataset("deflection", data = self.deflection_25D) # just always take note in the file name

        z = out_file.create_dataset("edge_moment", data = self.vomo.iloc[:, 2])
        z = out_file.create_dataset("edge_load", data = self.vomo.iloc[:, 1])
        out_file.attrs["stress_const"] = 1 # let's compute the bending moment for now!
        out_file.attrs["delta"] = self.delta
        out_file.attrs["rigidity"] = self.rigidity
        out_file.attrs["poisson"] = self.poisson

        out_file.close()

        print("Deflection values in: ", out_fname)

        return

    def solve_25D(self):

        x = np.flip(self.x)

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

