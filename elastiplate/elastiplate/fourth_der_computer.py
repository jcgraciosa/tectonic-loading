import numpy as np
import h5py

def compute_fourth_order(plate_fname, out_fname):

    indata = h5py.File(plate_fname, 'r')

    # read all of the contents of indata - easier for code revisions
    num_x_elem = indata['deflection'].shape[0]
    num_y_elem = indata['deflection'].shape[1]
    deflection = indata['deflection'][:, :]
    delta = indata.attrs['delta']
    rigidity = indata.attrs['rigidity']
    print('rigidity value: ', rigidity)
    p = indata.attrs['poisson']

    four_x = np.zeros([num_x_elem, num_y_elem])
    four_y = np.zeros([num_x_elem, num_y_elem])
    sec_x_sec_y = np.zeros([num_x_elem, num_y_elem])

    # stencils used here
    four_xy_sten = np.array([1, -4, 6, -4, 1])
    sec_x_sec_y_sten = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])

    for x in range(2, num_x_elem - 2):
        for y in range(2, num_y_elem - 2):

            # x is along rows
            four_x[x, y] = np.sum(deflection[x-2:x+3, y]*four_xy_sten)/(delta**4)
            four_y[x, y] = np.sum(deflection[x, y-2:y+3]*four_xy_sten)/(delta**4)
            sec_x_sec_y[x, y] = np.sum(deflection[x-1:x+2, y-1:y+2]*sec_x_sec_y_sten)/(delta**4)

            #if (x == 90) and (y == 60):
            #    print('der x: ', four_x[x, y])
            #    print('der y: ', four_y[x, y])
            #    print('der xy: ', sec_x_sec_y[x, y])


    four_x = rigidity*four_x
    four_y = rigidity*four_y
    sec_x_sec_y = 2*rigidity*sec_x_sec_y

    out_file = h5py.File(out_fname, 'w')
    out_file.attrs["delta"] = delta
    z = out_file.create_dataset("four_x", data = four_x)
    z = out_file.create_dataset("four_y", data = four_y)
    z = out_file.create_dataset("sec_x_sec_y", data = sec_x_sec_y)

    out_file.close()

    print("Output written to: ", out_fname)

    return

