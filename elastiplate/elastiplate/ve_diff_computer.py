import numpy as np
import h5py

def compute_ve_diff(delta_rho, gravity, deflect_3d_fname, deflect_2d_fname, out_fname, model2_params = None):

    # read the data and asign to local variables
    indata = h5py.File(deflect_2d_fname, "r")
    deflect_2d = indata["deflection"][:, :, :]

    indata = h5py.File(deflect_3d_fname, "r")
    deflect_3d = indata["deflection"][:, :, :]

    t_delta = indata.attrs["t_delta"]
    t_maxwell = indata.attrs["t_maxwell"]

    num_t_elem = deflect_3d.shape[0]
    num_x_elem = deflect_3d.shape[1]
    num_y_elem = deflect_3d.shape[2]

    # compute the time derivatives - so time_ky_val should start at index 1
    val_3d = np.zeros_like(deflect_3d)
    val_2d = np.zeros_like(deflect_2d)
    outval = np.zeros_like(deflect_2d)

    for i in np.arange(1, num_t_elem):
        dummy_3d = (deflect_3d[i, :, :] - deflect_3d[i-1, :, :])/t_delta
        dummy_2d = (deflect_2d[i, :, :] - deflect_2d[i-1, :, :])/t_delta

    #t_der_3d = (deflect_3d[1:] - deflect_3d[:-1])/t_delta
    #t_der_25d = (deflect_25d[1:] - deflect_25d[:-1])/t_delta

        if model2_params is None:
            val_3d[i, :, :] = -delta_rho*gravity*(t_maxwell*dummy_3d + deflect_3d[i, :, :])
            val_2d[i, :, :] = -delta_rho*gravity*(t_maxwell*dummy_2d + deflect_2d[i, :, :])
        """
        else:
            dummy_3d = -(4*model2_params["mantle_viscosity"]/model2_params["thickness"])
            dummy_3d = dummy_3d*t_der_3d*(6*model2_params["mantle_viscosity"]/model2_params["plate_viscosity"])**(1/3)
            val_3d = dummy_3d - delta_rho*gravity*deflect_3d[1:]

            dummy_2d = -(4*model2_params["mantle_viscosity"]/model2_params["thickness"])
            dummy_2d = dummy_2d*t_der_2d*(6*model2_params["mantle_viscosity"]/model2_params["plate_viscosity"])**(1/3)
            val_2d = dummy_2d - delta_rho*gravity*deflect_2d[1:]
        """
        outval[i, :, :] = val_3d[i, :, :] - val_2d[i, :, :]

    # pack output and save it
    # copy all data except the deflection into the output file
    out_file = h5py.File(out_fname, "w")
    z = out_file.create_dataset("val_3d", data = val_3d)
    z = out_file.create_dataset("val_2d", data = val_2d)
    z = out_file.create_dataset("difference", data = outval)

    z = out_file.create_dataset("Mo", data = indata["Mo"][:])
    z = out_file.create_dataset("Vo", data = indata["Vo"][:])
    z = out_file.create_dataset("Mo_rate", data = indata["Mo_rate"][:])
    z = out_file.create_dataset("Vo_rate", data = indata["Vo_rate"][:])
    z = out_file.create_dataset("x", data = indata["x"][:])
    z = out_file.create_dataset("y", data = indata["y"][:])
    z = out_file.create_dataset("time_ky", data = indata["time_ky"][1:])

    out_file.attrs["xy_delta"] =  indata.attrs["xy_delta"]
    out_file.attrs["t_delta"] =   indata.attrs["t_delta"]
    out_file.attrs["rigidity"] =  indata.attrs["rigidity"]
    out_file.attrs["poisson"] =   indata.attrs["poisson"]
    out_file.attrs["t_maxwell"] = indata.attrs["t_maxwell"]

    out_file.close()

    print("Difference values in: ", out_fname)

    return

