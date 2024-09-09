import numpy as np
import pandas as pd
import json, h5py

from scipy.linalg import det, inv, svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, lsqr
from scipy import interpolate

class Plate(object):


    def __init__(self, config_file):

        # unpack parameters in here
        with open(config_file) as json_data_file:
            conf = json.load(json_data_file)

        print(conf)
        conf_pl = conf['plate']

        self.rigidity = (conf_pl["modulus"]*conf_pl["thickness"]**3)/(12*(1-conf_pl["poisson"]**2))
        self.iso_term = ((conf_pl["rho_mantle"] - conf_pl["rho_infill"])*conf_pl["gravity"]*conf_pl["delta"]**4)/self.rigidity
        self.stress_const = 12*self.rigidity/conf_pl["thickness"]**3
        self.alpha = (4*self.rigidity/((conf_pl["rho_mantle"] - conf_pl["rho_infill"])*conf_pl['gravity']))**0.25

        self.num_x_elem = int(conf_pl["dim"][0]/conf_pl['delta']) + 1
        self.num_y_elem = int(conf_pl['dim'][1]/conf_pl['delta']) + 1

        self.x = np.arange(0, conf_pl["dim"][0] + conf_pl["delta"], conf_pl["delta"])
        self.y = np.arange(0, conf_pl['dim'][1] + conf_pl["delta"], conf_pl["delta"])

        self.delta = conf_pl["delta"] # copied for resolution test

        self._gen_loading(conf_pl)
        #self._solve_analytical(conf_pl, conf_ld, conf_mo)

        # initialize plate matrix
        self.a_matrix = np.zeros([self.num_x_elem*self.num_y_elem, self.num_x_elem*self.num_y_elem])

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


    def _gen_loading(self, conf_pl):

        self.load = np.zeros([self.num_x_elem, self.num_y_elem])
        self.edge_moment = np.zeros(self.num_y_elem)
        self.tot_moment = np.zeros([2, self.num_y_elem])

        # read vo and mo
        self.vomo_file = conf_pl['vomo_file']
        self.vomo = pd.read_csv(self.vomo_file)

        #print(self.vomo)


        # only supports load and moment specified at each point
        # transverse load in here
        self.load[-1, :] = self.vomo.iloc[:, 1]
        self.edge_field = (2*self.load*conf_pl['delta']**3)/self.rigidity

        # edge moment
        self.edge_moment[:] = self.vomo.iloc[:, 2]
        self.tot_moment[0, :] = (self.edge_moment*conf_pl['delta']**2)/self.rigidity
        self.tot_moment[1, :] = (-2-2*conf_pl['poisson'])*(self.edge_moment*conf_pl['delta']**2)/self.rigidity
        self.tot_moment[1, 2:-2] += conf_pl['poisson']*(self.edge_moment[3:-1]*conf_pl['delta']**2)/self.rigidity # east
        self.tot_moment[1, 2:-2] += conf_pl['poisson']*(self.edge_moment[1:-3]*conf_pl['delta']**2)/self.rigidity # west

        # edges
        self.tot_moment[1, 0] += 2*conf_pl['poisson']*self.edge_moment[1]*conf_pl['delta']**2/self.rigidity
        self.tot_moment[1, -1] += 2*conf_pl['poisson']*self.edge_moment[-2]*conf_pl['delta']**2/self.rigidity

        # penultimate
        self.tot_moment[1, 1] += conf_pl['poisson']*self.edge_moment[2]*conf_pl['delta']**2/self.rigidity
        self.tot_moment[1, -2] += conf_pl['poisson']*self.edge_moment[-3]*conf_pl['delta']**2/self.rigidity

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

        cp_stencil[2, 2] += self.iso_term
        #print(x, y)
        #print(cp_stencil)
        return cp_stencil

    def fill_matrix(self):

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
                self.a_matrix[arr_idx, idx_1d] = coeffs


    def solve_spsolve(self):

        # faster way to solve
        #inv_mat = inv(self.a_matrix)

        csr_mat = csr_matrix(self.a_matrix)
        self.deflection = spsolve(csr_mat, self.edge_field.reshape(self.num_x_elem*self.num_y_elem).T)
        self.deflection = -self.deflection.reshape([self.num_x_elem, self.num_y_elem])

        return

    def solve_lsqr(self):

        test_out = lsqr(self.a_matrix, self.edge_field.reshape(self.num_x_elem*self.num_y_elem))
        self.deflection = -test_out[0].reshape(self.num_x_elem, self.num_y_elem)

        return

    def solve_svd(self):

        U,s,Vh = svd(self.a_matrix)
        pinv_svd = np.dot(np.dot(Vh.T,inv(np.diag(s))),U.T)
        self.deflection = np.dot(pinv_svd, self.edge_field.reshape(self.num_x_elem*self.num_y_elem))
        self.deflection = -self.deflection.reshape(self.num_x_elem, self.num_y_elem)

        return

    def write_out(self, out_fname):

        out_file = h5py.File(out_fname, 'w')
        z = out_file.create_dataset("deflection", data = self.deflection)
        z = out_file.create_dataset("edge_moment", data = self.edge_moment)
        z = out_file.create_dataset("edge_load", data = self.load[-1, :])
        out_file.attrs["stress_const"] = self.stress_const
        out_file.attrs["delta"] = self.delta
        out_file.attrs["rigidity"] = self.rigidity
        out_file.attrs["poisson"] = self.poisson

        out_file.close()

        print("Deflection values in: ", out_fname)

        return

    """ UNUSED FUNCTIONS
    def _solve_analytical(self, conf_pl, conf_ld, conf_mo):

        if conf_ld["type"] == "const_edge" and conf_mo["type"] == "const_edge":
            self.alpha = (4*self.rigidity/((conf_pl["rho_mantle"] - conf_pl["rho_infill"])*conf_pl['gravity']))**0.25

            if self.alpha == 0:
                return

            V_o = conf_ld["param1"]
            M_o = conf_mo["param1"]
            x_prime = conf_pl["dim"][0] - self.x
            self.analytical = (self.alpha**2/(2*self.rigidity))*np.exp(-x_prime/self.alpha)*(-M_o*np.sin(x_prime/self.alpha) + (-V_o*self.alpha + M_o)*np.cos(x_prime/self.alpha))

        if conf_ld["type"] == "periodic":
            self.analytical = np.zeros(self.num_y_elem)
            row_st, row_ed = (np.array(conf_ld["param2"])//conf_pl['delta']).astype(int)
            wavelength = conf_ld["param4"]//conf_pl["delta"]
            x = np.array([x for x in range(row_st, row_ed+1)])

            def_amp = conf_ld["param1"]/(conf_pl["rho_infill"]*conf_pl["gravity"])
            def_amp = def_amp/((conf_pl["rho_mantle"]/conf_pl["rho_infill"])-1+((self.rigidity/(conf_pl["rho_infill"]*conf_pl["gravity"]))*(2*np.pi/conf_ld["param4"])**4))
            self.analytical[row_st:row_ed + 1] = def_amp*np.sin(2*np.pi*(x - row_st)/wavelength)

        return

    # function is no longer relevant and is only for 2D
    def compute_error(self):

        num_x_check = int(120e3/self.delta) + 1

        diff = self.analytical[num_x_check:] + self.deflection[num_x_check:, self.num_y_elem//2]
        diff = diff**2
        self.err = [diff.mean(), diff.std(), num_x_check]

        return

    def _gen_loading(self, conf_pl, conf_ld, conf_mo):

        self.load = np.zeros([self.num_x_elem, self.num_y_elem])
        self.edge_moment = np.zeros(self.num_y_elem)
        self.tot_moment = np.zeros([2, self.num_y_elem])

        # transverse load in here
        if  conf_ld["type"] in ["const_edge", "two_val_edge", "slab_pull", "two_val_slab_pull", "point_edge", "periodic_slab_pull"]:
            if conf_ld["type"] == "const_edge":
                self.load[-1, :] = conf_ld["param1"]
            elif conf_ld["type"] == "two_val_edge":
                lim_col = int(conf_ld["param3"]/conf_pl["delta"])
                self.load[-1, :lim_col] = conf_ld["param1"]
                self.load[-1, lim_col:] = conf_ld["param2"]
            elif conf_ld["type"] == "slab_pull":
                self.load[-1, :] = conf_ld["param1"]*conf_pl["gravity"]*conf_pl["thickness"]*conf_ld["param2"]
            elif conf_ld["type"] == "point_edge":
                idx = np.array(conf_ld["param2"])/conf_pl["delta"]
                idx = idx.astype(int)
                self.load[-1, idx] = conf_ld["param1"]
            elif conf_ld["type"] == "periodic_slab_pull":

                f = 2*np.pi/conf_pl["dim"][1]
                Lo = conf_ld["param2"]
                La = conf_ld["param3"]*4/np.pi
                coeffs = np.array([c for c in range(1000)])
                div = coeffs%2
                coeffs = coeffs[div != 0]

                y_ld = np.zeros_like(self.y)

                for i in coeffs:
                    y_prime = (1/i)*np.sin(i*f*self.y)
                    y_ld += y_prime

                #self.load[-1, :] = conf_ld["param1"]*conf_pl["gravity"]*conf_pl["thickness"]*(conf_ld["param2"] + conf_ld["param3"]*np.sin(2*np.pi*self.y/conf_ld["param4"]))
                self.load[-1, :] = conf_ld["param1"]*conf_pl["gravity"]*conf_pl["thickness"]*(conf_ld["param2"] + conf_ld["param3"]*4*y_ld/np.pi)
            else: # two_val_slab_pull
                lim_col = int(conf_ld["param4"]/conf_pl["delta"])
                self.load[-1, :lim_col] = conf_ld["param1"]*conf_pl["gravity"]*conf_pl["thickness"]*conf_ld["param2"]
                self.load[-1, lim_col:] = conf_ld["param1"]*conf_pl["gravity"]*conf_pl["thickness"]*conf_ld["param3"]

            self.edge_field = (2*self.load*conf_pl['delta']**3)/self.rigidity

            #self.edge_field[-1, 0] += (4*self.load[-1, 0]*conf_pl['delta']**2)/self.rigidity
            #self.edge_field[-1, -1] += (4*self.load[-1, -1]*conf_pl['delta']**2)/self.rigidity
            #self.edge_field[-1, 0] += (2*self.load[-1, 0]*conf_pl['delta']**3)/self.rigidity
            #self.edge_field[-1, -1] += (2*self.load[-1, -1]*conf_pl['delta']**3)/self.rigidity

        elif conf_ld["type"] == "uniform":
            row_st, row_ed = (np.array(conf_ld["param2"])//conf_pl['delta']).astype(int)
            col_st, col_ed = (np.array(conf_ld["param3"])//conf_pl['delta']).astype(int)

            self.load[row_st: row_ed+1, col_st: col_ed+1] = conf_ld["param1"]
            self.edge_field = self.load*conf_pl['delta']**4/self.rigidity

        elif conf_ld["type"] == "periodic":
            amp = conf_ld["param1"]
            row_st, row_ed = (np.array(conf_ld["param2"])//conf_pl['delta']).astype(int)
            col_st, col_ed = (np.array(conf_ld["param3"])//conf_pl['delta']).astype(int)
            wavelength = conf_ld["param4"]//conf_pl['delta']

            x = np.array([x for x in range(row_st, row_ed+1)])

            self.load[row_st: row_ed+1, col_st: col_ed+1] = np.repeat(amp*np.sin(2*np.pi*(x - row_st)/wavelength), col_ed-col_st+1).reshape(row_ed-row_st+1, col_ed-col_st+1)
            self.edge_field = self.load*conf_pl['delta']**4/self.rigidity

        elif conf_ld["type"] == "function":

            #row = (np.array(conf_ld["param2"])//conf_pl['delta']).astype(int)
            #col = (np.array(conf_ld["param3"])//conf_pl['delta']).astype(int)
            #print(row)
            #print(col)
            #print(self.load.shape)
            #self.load[row, col] = np.array(conf_ld["param1"])

            y_val = np.array(conf_ld["param3"]) - 70e3
            f_v = interpolate.interp1d(y_val, np.array(conf_ld["param1"]))
            ynew = np.arange(70e3, 450e3, 2e3)
            vo_new = f_v(ynew)
            col = (ynew//conf_pl['delta']).astype(int)

            self.load[-1, col] = vo_new
            self.edge_field = (2*self.load*conf_pl['delta']**3)/self.rigidity

        # edge moments in here
        if conf_mo["type"] == "const_edge":
            self.edge_moment[:] = conf_mo["param1"]

        elif conf_mo["type"] == "two_val_edge":
            lim_col = int(conf_mo["param3"]/conf_pl["delta"])
            self.edge_moment[:lim_col] = conf_mo["param1"]
            self.edge_moment[lim_col:] = conf_mo["param2"]

        elif conf_mo["type"] == "point_edge":
            idx = np.array(conf_mo["param2"])/conf_pl["delta"]
            idx = idx.astype(int)
            self.edge_moment[idx] = conf_mo["param1"]

        elif conf_mo["type"] == "function":

            y_val = np.array(conf_ld["param3"]) - 70e3
            f_m = interpolate.interp1d(y_val, np.array(conf_mo["param1"]))
            ynew = np.arange(70e3, 450e3, 2e3)
            mo_new = f_m(ynew)
            col = (ynew//conf_pl['delta']).astype(int)

            self.edge_moment[col] = mo_new # exclude corners

        elif conf_mo["type"] == "periodic_slab_pull":

            amp = conf_mo["param1"]*conf_pl["gravity"]*conf_pl["thickness"] #delta_rho*g*h
            # try to represent load as series
            '''
            Lo = conf_mo["param2"]
            La = 4*conf_mo["param3"]/np.pi
            f = 2*np.pi/conf_pl["dim"][1]

            # odd and even coefficients
            num_coeff = 2000
            odd_coeffs = np.array([c for c in range(num_coeff)])
            div = odd_coeffs%2
            odd_coeffs = odd_coeffs[div != 0]

            even_coeffs = np.array([c for c in range(num_coeff)])
            div = even_coeffs%2
            even_coeffs = even_coeffs[div == 0] + 2

            # term 1
            sqlen1 = Lo**2 + ((La*np.pi)**2)/12

            # term 2
            y_dummy = np.zeros_like(self.y)
            for i in odd_coeffs:
                y_prime = (1/i)*np.sin(i*f*self.y)
                y_dummy += y_prime

            sqlen2 = 2*Lo*La*y_dummy

            # term 3
            y_dummy = np.zeros_like(self.y)
            for i in odd_coeffs:
                y_prime = (1/i**2)*np.cos(2*i*f*self.y)

                y_dummy += y_prime

            sqlen3 = 0.5*(La**2)*y_dummy

            # term 4 and 5
            y_dummy = np.zeros_like(self.y)
            for i in odd_coeffs:
                for j in even_coeffs:
                    const = 1/(i*(i + j))
                    y2 = const*(np.cos(j*f*self.y) - np.cos((2*i + j)*f*self.y))

                    y_dummy += y2

            sqlen45 = (La**2)*y_dummy

            self.edge_moment[:] = -amp*(sqlen1 + sqlen2 - sqlen3 + sqlen45) # hopefully this is the last '''

            self.edge_moment[:] = -amp*(conf_mo["param2"] + conf_mo["param3"]*4*y_ld/np.pi)**2
            #self.edge_moment[:] = -amp*(conf_mo["param2"]**2 + 0.5*conf_mo["param3"]**2)
            #self.edge_moment[:] += -2*amp*conf_mo["param2"]*conf_mo["param3"]*np.sin(2*np.pi*self.y/conf_mo["param4"]) #FIXME: add cos part
            #self.edge_moment[:] += 0.5*amp*(conf_mo["param3"]**2)*np.cos(4*np.pi*self.y/conf_mo["param4"])

        elif conf_mo["type"] == "load_dep":
            if conf_ld["type"] not in ["const_edge", "slab_pull"]:
                print("Unsupported end load and moment combination!!!") # FIXME: replace error message
                return
            else:
                self.edge_moment[:] = self.load[-1, :]*conf_mo["param1"]

        elif conf_mo["type"] == "two_val_load_dep":
            if conf_ld["type"] not in ["two_val_edge", "two_val_slab_pull"]:
                print("Unsupported end load and moment combination!!!") # FIXME: replace error message
                return
            else:
                self.edge_moment[:lim_col] = self.load[-1, :lim_col]*conf_mo["param1"]
                self.edge_moment[lim_col:] = self.load[-1, lim_col:]*conf_mo["param2"]

        self.tot_moment[0, :] = (self.edge_moment*conf_pl['delta']**2)/self.rigidity
        self.tot_moment[1, :] = (-2-2*conf_pl['poisson'])*(self.edge_moment*conf_pl['delta']**2)/self.rigidity
        self.tot_moment[1, 2:-2] += conf_pl['poisson']*(self.edge_moment[3:-1]*conf_pl['delta']**2)/self.rigidity # east
        self.tot_moment[1, 2:-2] += conf_pl['poisson']*(self.edge_moment[1:-3]*conf_pl['delta']**2)/self.rigidity # west

        # edges
        self.tot_moment[1, 0] += 2*conf_pl['poisson']*self.edge_moment[1]*conf_pl['delta']**2/self.rigidity
        self.tot_moment[1, -1] += 2*conf_pl['poisson']*self.edge_moment[-2]*conf_pl['delta']**2/self.rigidity

        # penultimate
        self.tot_moment[1, 1] += conf_pl['poisson']*self.edge_moment[2]*conf_pl['delta']**2/self.rigidity
        self.tot_moment[1, -2] += conf_pl['poisson']*self.edge_moment[-3]*conf_pl['delta']**2/self.rigidity

        self.edge_field[-2:,:] += self.tot_moment


        return


    """


