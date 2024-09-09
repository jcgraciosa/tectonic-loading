import numpy as np
import h5py


def compute_stresses(plate_fname, out_fname):

    indata = h5py.File(plate_fname, 'r')

    # read all of the contents of indata - easier for code revisions
    num_x_elem = indata['deflection'].shape[0]
    num_y_elem = indata['deflection'].shape[1]
    deflection = indata['deflection'][:, :]
    edge_moment = indata['edge_moment'][:]
    stress_const = indata.attrs['stress_const']
    delta = indata.attrs['delta']
    rigidity = indata.attrs['rigidity']
    p = indata.attrs['poisson']
    p2 = p**2

    m_x = np.zeros([num_x_elem + 1, num_y_elem])
    m_y = np.zeros([num_x_elem + 1, num_y_elem])
    m_xy = np.zeros([num_x_elem + 1, num_y_elem])

    deflection = np.vstack([np.zeros(num_y_elem), deflection])

    # stencils used here
    mx_sten = np.array([[0, 1, 0], [p, -2-2*p, p], [0, 1, 0]])
    my_sten = np.array([[0, p, 0], [1, -2-2*p, 1], [0, p, 0]])
    mxy_sten = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    # mxy stencils
    mxy_r1_l = np.array([[-p, 0], [-2-2*p, 2], [p, 0]])
    mxy_r1_r = -np.fliplr(mxy_r1_l)
    mxy_r2up_l = np.array([[-p, 0], [2+2*p, -2], [0, 0], [-2-2*p, 2], [p, 0]])
    mxy_r2up_r = -np.fliplr(mxy_r2up_l)
    mxy_rpen_l = np.array([[-p, 0], [2+2*p, -2], [-p, 0], [-2, 2]])
    mxy_rpen_r = -np.fliplr(mxy_rpen_l)
    mxy_rl_l = np.array([[2, 0, -2, 0], [-2, -p, 2+2*p, -p]])
    mxy_rl_r = -np.fliplr(mxy_rl_l)
    mxy_rl = np.array([[0, 2, 0, -2, 0], [p, -2-2*p, 0, 2+2*p, -p]])

    # separate computation of m_x and m_y from m_xy - m_xy is such a pain in the ass
    for x in range(1, num_x_elem + 1):
        for y in range(num_y_elem):
            if x == 1: # note that x == 0 is just a dummy
                if y in [0, num_y_elem -1]:
                    m_y[x, y] = 0
                    m_x[x, y] = -rigidity*(1-p2)*(-2*deflection[x, y] + deflection[x+1, y])/delta**2
                else:
                    m_x[x, y] = -rigidity*np.sum(deflection[x-1:x+2, y-1:y+2]*mx_sten)/delta**2
                    m_y[x, y] = -rigidity*np.sum(deflection[x-1:x+2, y-1:y+2]*my_sten)/delta**2
            elif x == num_x_elem:
                if y in [0, num_y_elem -1]:
                    m_x[x, y] = edge_moment[y]
                    m_y[x, y] = 0
                else:
                    m_x[x, y] = edge_moment[y]
                    m_y[x, y] = -rigidity*(1-p2)*(deflection[x, y-1] - 2*deflection[x, y] + deflection[x, y+1])/delta**2
                    m_y[x, y] += p*edge_moment[y]
            else:
                if y in [0, num_y_elem -1]:
                    m_y[x, y] = 0
                    m_x[x, y] = -rigidity*(1-p2)*(deflection[x-1,y] - 2*deflection[x, y] + deflection[x+1, y])/delta**2
                else:
                    m_x[x, y] = -rigidity*np.sum(deflection[x-1:x+2, y-1:y+2]*mx_sten)/delta**2
                    m_y[x, y] = -rigidity*np.sum(deflection[x-1:x+2, y-1:y+2]*my_sten)/delta**2

    # computation of m_xy
    for x in range(1, num_x_elem + 1):
        for y in range(num_y_elem):
            if x == 1:
                if y == 0:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x:x+3, y:y+2]*mxy_r1_l)
                elif y == num_y_elem - 1:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x:x+3, y-1:y+1]*mxy_r1_r)
                else:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-1:x+2, y-1:y+2]*mxy_sten)
            elif x == num_x_elem - 1:
                if y == 0:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-2:x+2, y:y+2]*mxy_rpen_l) - p*m_x[x+1, y]/(4*(1+p))
                elif y == num_y_elem - 1:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-2:x+2, y-1:y+1]*mxy_rpen_r) + p*m_x[x+1, y]/(4*(1+p))
                else:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-1:x+2, y-1:y+2]*mxy_sten)
            elif x == num_x_elem:
                if y in [0, num_y_elem - 1]:
                    m_xy[x, y] = 0
                elif y == 1:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-1:x+1, y-1:y+3]*mxy_rl_l)
                    m_xy[x, y] += m_x[x, y-1]/(4*(1+p)) - (1-p)*m_x[x, y+1]/4 - p*m_y[x, y-1]/(4*(1+p))
                elif y == num_y_elem - 2:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-1:x+1, y-2:y+2]*mxy_rl_r)
                    m_xy[x, y] += -m_x[x, y+1]/(4*(1+p)) + (1-p)*m_x[x, y-1]/4 + p*m_y[x, y+1]/(4*(1+p))
                else:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-1:x+1, y-2:y+3]*mxy_rl)
                    m_xy[x, y] += (1-p)*(m_x[x, y-1] - m_x[x, y+1])/4
            else:
                if y == 0:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-2:x+3, y:y+2]*mxy_r2up_l)
                elif y == num_y_elem - 1:
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-2:x+3, y-1:y+1]*mxy_r2up_r)
                else:
                    #print(x, y)
                    #print('deflection shape', deflection[x-1:x+2, y-1:y+2].shape)
                    #print('mxy shape',mxy_sten.shape)
                    m_xy[x, y] = (rigidity*(1-p)/(4*delta**2))*np.sum(deflection[x-1:x+2, y-1:y+2]*mxy_sten)

    sigma_x = -stress_const*m_x[1:, :]
    sigma_y = -stress_const*m_y[1:, :]
    sigma_xy = -stress_const*m_xy[1:, :]
    m_x = m_x[1:, :]
    m_y = m_y[1:, :]
    m_xy = m_xy[1:, :]
    deflection = deflection[1:, :]

    #save the stresses and bending moments
    out_file = h5py.File(out_fname, 'w')
    z = out_file.create_dataset("sigma_x", data = sigma_x)
    z = out_file.create_dataset("sigma_y", data = sigma_y)
    z = out_file.create_dataset("sigma_xy", data = sigma_xy)

    z = out_file.create_dataset("m_x", data = m_x)
    z = out_file.create_dataset("m_y", data = m_y)
    z = out_file.create_dataset("m_xy", data = m_xy)

    out_file.attrs["delta"] = delta
    out_file.attrs['stress_const'] = stress_const
    out_file.close()

    print("Output written to: ", out_fname)

    return
