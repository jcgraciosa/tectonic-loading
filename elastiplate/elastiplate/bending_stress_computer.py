import numpy as np
import h5py

def compute_bending_stresses(plate_fname, out_fname, thickness = 10e3):

    indata = h5py.File(plate_fname, 'r')

    # read all of the contents of indata - easier for code revisions
    num_x_elem = indata['deflection'].shape[0]
    num_y_elem = indata['deflection'].shape[1]
    deflection = indata['deflection'][:, :]
    delta = indata.attrs['delta']
    rigidity = indata.attrs['rigidity']
    print('rigidity value: ', rigidity)
    p = indata.attrs['poisson']

    half_thk = thickness/2
    const_xx_yy = -12*rigidity*half_thk/(thickness**3)
    const_xy = const_xx_yy*(1-p)

    sec_der_x = np.zeros([num_x_elem, num_y_elem])
    sec_der_y = np.zeros([num_x_elem, num_y_elem])
    der_x_der_y = np.zeros([num_x_elem, num_y_elem])

    # stencils used here
    derder_sten = np.array([1, -2, 1])
    der_xy_sten = np.array([[1, 0, -1], 
                            [0, 0, 0], 
                            [-1, 0, 1]]) # original: [1, 0, -1]

    for x in range(1, num_x_elem - 1):
        for y in range(1, num_y_elem - 1):

            sec_der_x[x, y] = np.sum(deflection[x-1:x+2, y]*derder_sten)/(delta**2)
            sec_der_y[x, y] = np.sum(deflection[x, y-1:y+2]*derder_sten)/(delta**2)
            der_x_der_y[x, y] = np.sum(deflection[x-1:x+2, y-1:y+2]*der_xy_sten)/(4*delta**2)

    sigma_xx = const_xx_yy*(sec_der_x + p*sec_der_y)
    sigma_yy = const_xx_yy*(sec_der_y + p*sec_der_x)
    sigma_xy = const_xy*der_x_der_y

    out_file = h5py.File(out_fname, 'w')
    out_file.attrs["delta"] = delta
    out_file.attrs["const_xx_yy"] = const_xx_yy
    out_file.attrs["const_xy"] = const_xy
    z = out_file.create_dataset("sigma_xx", data = sigma_xx)
    z = out_file.create_dataset("sigma_yy", data = sigma_yy)
    z = out_file.create_dataset("sigma_xy", data = sigma_xy)

    out_file.close()

    print("Output written to: ", out_fname)

    return

