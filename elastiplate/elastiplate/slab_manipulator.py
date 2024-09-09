import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.signal import medfilt
from scipy.interpolate import SmoothBivariateSpline


class SlabManipulator(object):

    def __init__(self, slab_xyz, trench_xy, earth_rad_km):

        self.trench_df = pd.read_csv(trench_xy, sep='\t', header = None)
        self.slab_df = pd.read_csv(slab_xyz, header = None)
        self.slab_df = self.slab_df[~self.slab_df[2].isnull()] # remove null values

        self.earth_radius = earth_rad_km

        self.lon_reso = 0.05
        self.lat_reso = 0.05 # is this the same for all slab data?

        return

    ##### INTERNAL FUNCTIONS #####
    def _haversine_dist(self, st_lonlat1, st_lonlat2):

        lon1 = st_lonlat1[0]*np.pi/180
        lat1 = st_lonlat1[1]*np.pi/180
        lon2 = st_lonlat2[0]*np.pi/180
        lat2 = st_lonlat2[1]*np.pi/180

        a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2
        c = 2*np.arcsin(np.min([1, np.sqrt(a)]))

        return self.earth_radius*c

    def _lonlat_to_colrow(self, lon, lat):

        row = 10 + ((lat - self.slab_df[1].min())/self.lat_reso).astype(int) # lat
        col = 10 + ((lon - self.slab_df[0].min())/self.lon_reso).astype(int) # lon

        return np.vstack([col, row]).T

    def _get_dist_along_track(self, start_lonlat, lonlat):

        distance = np.zeros(lonlat.shape[0])

        st_lonlat = start_lonlat
        for idx, xy in zip(np.arange(len(distance)), lonlat):

            distance[idx] = self._haversine_dist(st_lonlat, xy)
            st_lonlat = xy

        return np.cumsum(distance)

    ##### EXTERNAL FUNCTIONS #####
    def get_slab_border(self):

        lon_num = 20 + int((self.slab_df[0].max() + self.lon_reso - self.slab_df[0].min())/self.lon_reso) + 1
        lat_num = 20 + int((self.slab_df[1].max() + self.lat_reso - self.slab_df[1].min())/self.lat_reso) + 1

        self.slab_grd = np.zeros([lat_num, lon_num])

        # Make a function from this?
        colrow = self._lonlat_to_colrow(np.asarray(self.slab_df[0]), np.asarray(self.slab_df[1]))
        col = colrow[:, 0]
        row = colrow[:, 1]

        self.slab_grd[row, col] = 255
        self.slab_grd = self.slab_grd.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        self.border_img = cv2.morphologyEx(self.slab_grd, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(self.border_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.cim = np.zeros_like(self.border_img)
        cv2.drawContours(self.cim, contours, -1, 255, 1)

        row_nonzero, col_nonzero = np.where(self.cim > 0)
        lat_inv = (self.slab_df[1].min() + (row_nonzero - 10)*self.lat_reso)
        lon_inv = (self.slab_df[0].min() + (col_nonzero - 10)*self.lon_reso)

        # assign to what is available in the slab data
        blank_x = []
        blank_y = []

        x_bndry = []
        y_bndry = []
        z_bndry = []

        for x, y in zip(lon_inv.round(2), lat_inv.round(2)):
            dum = self.slab_df[(self.slab_df[0] == x) & (self.slab_df[1] == y)]
            dum = np.asarray(dum)

            if dum.shape[0] == 0:
                blank_x.append(x)
                blank_y.append(y)
            else:
                x_bndry.append(dum[0, 0])
                y_bndry.append(dum[0, 1])
                z_bndry.append(dum[0, 2])

        # assign to nearest - for data that have no exact match
        conv_mat = np.asmatrix(self.slab_df)
        for x, y in zip(blank_x, blank_y):
            dist = np.sqrt(np.power(x - self.slab_df[0], 2) + np.power(y - self.slab_df[1], 2))
            least = np.where(dist == dist.min())[0]
            dum = conv_mat[least]
            x_bndry.append(dum[0, 0])
            y_bndry.append(dum[0, 1])
            z_bndry.append(dum[0, 2])

        self.x_bndry = np.asarray(x_bndry)
        self.y_bndry = np.asarray(y_bndry)
        self.z_bndry = np.asarray(z_bndry)

        return

    def make_histogram(self, thresh_value):

        count, hist_x = np.histogram(self.z_bndry[self.z_bndry > thresh_value], bins = 'fd')

        fig = plt.figure(dpi = 300)
        plt.bar(hist_x[:-1] + np.mean(hist_x[1] - hist_x[0]), count, align = 'center', edgecolor = 'black')

        return


    def apply_threshold(self, thresh_value):

        self.trench_x = self.x_bndry[self.z_bndry > -20]
        self.trench_y = self.y_bndry[self.z_bndry > -20]
        self.trench_z = self.z_bndry[self.z_bndry > -20]

        return

    def assign_trench_from_file(self):

        self.trench_x = np.asarray(self.trench_df[0]) # longitude
        self.trench_y = np.asarray(self.trench_df[1]) # latitude
        self.trench_z = np.zeros_like(self.trench_y) # just dummy values in here

        return

    def order_trench(self, st_lonlat, rm_ends):

        st_lon = st_lonlat[0]
        st_lat = st_lonlat[1]

        ordered_x = np.zeros(self.trench_x.shape)
        ordered_y = np.zeros(self.trench_x.shape)
        ordered_z = np.zeros(self.trench_x.shape)

        # FIXME: convert this to spherical coordinates - do you need to do this?
        # find nearest to st_latlon
        dist = np.sqrt(np.power(st_lon - self.trench_x, 2) + np.power(st_lat - self.trench_y, 2))
        least = np.where(dist == dist.min())[0]
        ordered_x[0], ordered_y[0], ordered_z[0] = self.trench_x[least], self.trench_y[least], self.trench_z[least]

        cp_x = np.delete(self.trench_x, least)
        cp_y = np.delete(self.trench_y, least)
        cp_z = np.delete(self.trench_z, least)
        for idx in range(1, self.trench_x.shape[0]):
            dist = np.sqrt(np.power(ordered_x[idx - 1] - cp_x, 2) + np.power(ordered_y[idx - 1] - cp_y, 2))
            least = np.where(dist == dist.min())[0]
            if least.shape[0] > 1:
                least = least[0]
            ordered_x[idx], ordered_y[idx], ordered_z[idx] = cp_x[least], cp_y[least], cp_z[least]
            cp_x = np.delete(cp_x, least)
            cp_y = np.delete(cp_y, least)
            cp_z = np.delete(cp_z, least)

        # remove some parts
        st_rm = rm_ends[0]
        ed_rm = -rm_ends[1]
        if ed_rm < 0:
            self.trench_x = ordered_x[st_rm:ed_rm]
            self.trench_y = ordered_y[st_rm:ed_rm]
            self.trench_z = ordered_z[st_rm:ed_rm]
        else:
            self.trench_x = ordered_x[st_rm:]
            self.trench_y = ordered_y[st_rm:]
            self.trench_z = ordered_z[st_rm:]

        return

    def get_norm_vecs(self, filt_width):

        plus1_x = self.trench_x[2:]
        minus1_x = self.trench_x[:-2]
        plus1_y = self.trench_y[2:]
        minus1_y = self.trench_y[:-2]

        der1 = (plus1_y - minus1_y)/(plus1_x - minus1_x)
        angle = np.arctan(der1)
        angle_filt = medfilt(angle, filt_width)

        x_vec = np.cos(angle_filt)
        y_vec = np.sin(angle_filt)

        # normal vectors - based on rotation matrices
        self.norm1 = np.vstack([-y_vec, x_vec]).T
        self.norm2 = np.vstack([y_vec, -x_vec]).T

        # select which norm to is pointing towards slab
        idx = 0
        self.final_norm = np.zeros_like(self.norm1)
        for x, y in zip(self.trench_x[1:-1], self.trench_y[1:-1]):

            # check norm1 in here
            x_prime = x + self.norm1[idx, 0]
            y_prime = y + self.norm1[idx, 1]
            colrow = self._lonlat_to_colrow(x_prime, y_prime)
            col = colrow[:, 0]
            row = colrow[:, 1]

            if row >= self.border_img.shape[0] or col >= self.border_img.shape[1]:
                self.final_norm[idx, :] = self.norm2[idx, :]
            else:
                if (self.border_img[row, col] == 255):
                    self.final_norm[idx, :] = self.norm1[idx, :]
                else:
                    self.final_norm[idx, :] = self.norm2[idx, :]

            idx += 1

        return

    def get_dist_along_trench(self):

        lonlat = np.vstack([self.trench_x[1:-1], self.trench_y[1:-1]]).T
        self.dist_along_trench = self._get_dist_along_track([self.trench_x[1], self.trench_y[1]], lonlat)

        return

    def sample_along_trench(self, step):

        self.idx_list = []
        self.idx_list.append(0) # 0 value

        curr = step
        offset = 0
        cp_list = np.copy(self.dist_along_trench)
        while(curr <= self.dist_along_trench[-1]):

            lin_dist = np.abs(cp_list - curr)
            idx = np.where(lin_dist == lin_dist.min())[0][0] # what if repeating?
            self.idx_list.append(offset + idx)

            curr += step
            offset = offset + idx + 1
            cp_list = cp_list[idx + 1:]
            if(offset + idx >= self.dist_along_trench.shape[0]): # make sure that once end is reached, we are done
                break

        self.idx_list = np.array(self.idx_list)

        return

    def sample_along_normals(self, vec_len):

        # convert sample points along trench to colrow
        st_xy = self._lonlat_to_colrow(self.trench_x[self.idx_list+1], self.trench_y[self.idx_list+1])
        ed_xy = self._lonlat_to_colrow(self.trench_x[self.idx_list+1] + vec_len*self.final_norm[self.idx_list, 0],
                                       self.trench_y[self.idx_list+1] + vec_len*self.final_norm[self.idx_list, 1])

        conv_mat = np.asmatrix(self.slab_df)
        self.samples = {}

        # use self.idx_list since when using get_all_points, we are
        # sampling the distance along trench
        for idx, st, ed in zip(self.idx_list, st_xy, ed_xy):
            img = np.zeros_like(self.border_img)
            cv2.line(img, tuple(st), tuple(ed), (255, 255, 255))
            out = self.border_img & img

            row_nonzero, col_nonzero = np.where(out > 0)
            # re-order them according to distance from the st - st is in colrow
            dist = np.sqrt(np.power(st[0] - col_nonzero, 2) + np.power(st[1] - row_nonzero, 2))
            row_nonzero = row_nonzero[np.argsort(dist)]
            col_nonzero = col_nonzero[np.argsort(dist)]

            lat_inv = (self.slab_df[1].min() + (row_nonzero - 10)*self.lat_reso)
            lon_inv = (self.slab_df[0].min() + (col_nonzero - 10)*self.lon_reso)

            x_prof = np.zeros(row_nonzero.shape)
            y_prof = np.zeros(row_nonzero.shape)
            z_prof = np.zeros(row_nonzero.shape)

            i = 0
            for x, y in zip(lon_inv.round(2), lat_inv.round(2)):
                dum = self.slab_df[(self.slab_df[0] == x) & (self.slab_df[1] == y)]
                dum = np.asarray(dum)

                if dum.shape[0] == 0:
                    continue

                # if nothing is found then skip it
                #if dum.shape[0] == 0:
                #    dist = np.sqrt(np.power(x - self.slab_df[0], 2) + np.power(y - self.slab_df[1], 2))
                #    least = np.where(dist == dist.min())[0]
                #    dum = conv_mat[least]

                x_prof[i] = dum[0, 0]
                y_prof[i] = dum[0, 1]
                z_prof[i] = dum[0, 2]
                i += 1

            min_idx = np.where(z_prof == z_prof.min())[0][0] + 1 # deepest
            x_prof = x_prof[:min_idx]
            y_prof = y_prof[:min_idx]
            z_prof = z_prof[:min_idx]
            z_prof = z_prof - z_prof.max() # so that everything starts at 0

            dist = self._get_dist_along_track([x_prof[0], y_prof[0]], np.vstack([x_prof, y_prof]).T)
            self.samples[idx] = np.vstack([x_prof, y_prof, z_prof, dist]).T

        return

    def interpolate_along_dip(self, reso, interp = 'quadratic'): # this should work on the samples data and will have the same format as samples

        self.dip_interpolated = {}
        self.col_size = np.zeros(self.dist_along_trench.shape)
        self.reso = reso

        for i, idx in zip(np.arange(self.dist_along_trench.shape[0]), self.samples):
            prof = self.samples[idx]
            orig_dist = np.asarray(prof[:, 3])
            orig_depth = np.asarray(prof[:, 2])
            #print(orig_dist)
            f = interpolate.interp1d(orig_dist, orig_depth, kind = interp)
            xnew = np.arange(0, orig_dist[-1], self.reso)
            ynew = f(xnew)

#             if xnew[-1] > orig_dist[-1]: # last one # don't add the last one for simplicity
#                 xnew = np.append(xnew, orig_dist[-1])
#                 ynew = np.append(ynew, orig_depth[-1])

            self.col_size[i] = xnew.shape[0]

            zero_pad = np.zeros_like(xnew)

            self.dip_interpolated[idx] = np.vstack([zero_pad, zero_pad, ynew, xnew]).T

        return

    def compute_second_der(self, depth_lim = -100):

        nrow = int(self.col_size.max())
        ncol = self.dist_along_trench[self.idx_list].shape[0]

        self.grid_depth =np.ones([nrow, ncol])
        self.der_x = np.zeros([nrow, ncol]) # along dip
        self.der_y = np.zeros([nrow, ncol]) # along strike

        for j, idx in zip(np.arange(ncol), self.dip_interpolated):
            depth = self.dip_interpolated[idx][:, 2]
            depth = depth[depth >= depth_lim]

            self.grid_depth[:depth.shape[0], j] = depth

        print(self.grid_depth)

        # compute the second derivative
        # along dip (x) - simple since the resolution is uniform
        for j in np.arange(ncol):
            prof = self.grid_depth[:, j]
            prof_0 = prof[1:-1]
            prof_m1 = prof[:-2]
            prof_p1 = prof[2:]
            der = (prof_m1 + 2*prof_0 + prof_p1)/self.reso
            self.der_x[1:-1, j] = der

        # along strike/trench (y)
        dummy = np.copy(self.dist_along_trench[self.idx_list])
        dum_0 = dummy[1:-1]
        dum_m1 = dummy[:-2]
        dum_p1 = dummy[2:]
        coeffs = np.vstack([2/((dum_0-dum_m1)*(dum_p1-dum_m1)), -2/((dum_p1-dum_0)*(dum_0-dum_m1)), 2/((dum_p1-dum_0)*(dum_p1-dum_m1))]).T
        for i in np.arange(nrow):
            prof = self.grid_depth[i, :]
            vals = np.vstack([prof[:-2], prof[1:-1], prof[2:]]).T
            der = (coeffs*vals).sum(axis = 1)
            self.der_y[i, 1:-1] = der

        return


    def get_all_points(self, to_get):

        self.all_points = []

        for idx in to_get:
            along_trench = self.dist_along_trench[idx]
            prof = to_get[idx]
            for dist, depth in zip(prof[:, 3], prof[:, 2]):
                self.all_points.append([along_trench, dist, depth])

        self.all_points = np.array(self.all_points)
        # column 0 - along trench
        # column 1 - along dip
        # column 3 - depth

        return

    def get_until_depth(self, to_limit, max_depth = -100): # to_limit should have the same format as samples

        self.depth_limited = []

        for idx in to_limit:
            along_trench = self.dist_along_trench[idx]
            prof = to_limit[idx]
            for dist, depth in zip(prof[:, 3], prof[:, 2]):
                self.depth_limited.append([along_trench, dist, depth])
                if depth < max_depth: # since depth is negative
                    break

        self.depth_limited = np.array(self.depth_limited)
        # column 0 - along trench
        # column 1 - along dip
        # column 3 - depth

        return

    def compute_slab_prop(self, to_proc, st_depth, delta_rho, thickness, gravity = 9.82):

        self.slab_length = []

        for idx in to_proc:
            along_trench = self.dist_along_trench[idx]

            x = to_proc[idx][:, 3]
            z = to_proc[idx][:, 2]
            x = x[z < st_depth]
            z = z[z < st_depth]

            x1 = x[:-1]
            x2 = x[1:]
            z1 = z[:-1]
            z2 = z[1:]
            dist = np.sqrt((x1 - x2)**2 + (z1 - z2)**2)
            dist = np.sum(dist)
            self.slab_length.append([along_trench, dist*1e3])

        self.slab_length = np.array(self.slab_length)
        self.slab_pull = np.copy(self.slab_length)
        self.slab_pull[:, 1] = delta_rho*gravity*thickness*1e3*self.slab_length[:, 1]

        return

#     # I think this should not be used
#     def interpolate_on_grid(self, grid_space, deg): # grid space is in km

#         max_along_trench = self.all_points[:, 0].max()
#         max_along_dip = self.all_points[:, 1].max()

#         trench = np.arange(0, max_along_trench, grid_space)
#         dip = np.arange(0, max_along_dip, grid_space)

#         self.grid_trench, self.grid_dip = np.meshgrid(trench, dip)

#         self.grid_trench = self.grid_trench.flatten()
#         self.grid_dip = self.grid_dip.flatten()

#         for trench_val in trench:
#             dist = np.abs(trench_val - self.all_points[:, 0])
#             least = np.where(dist == dist.min())[0][0]
#             nearest_trench = self.all_points[least, 0]
#             dip_idx = np.where(self.all_points[:, 0] == nearest_trench)
#             dip_val = self.all_points[dip_idx, 1]
#             max_dip = dip_val.max()

#             del_idx = np.where((self.grid_trench == trench_val) & (self.grid_dip > max_dip))[0]
#             self.grid_trench = np.delete(self.grid_trench, del_idx)
#             self.grid_dip = np.delete(self.grid_dip, del_idx)

#         interpolator = SmoothBivariateSpline(self.all_points[:, 0], self.all_points[:, 1], self.all_points[:, 2], kx = deg, ky = deg)

#         self.grid_depth = interpolator.ev(self.grid_trench, self.grid_dip)

#         return

