# this script makes synthetic elevation data from a prescribed basal vertical
# velocity anomaly. the forward operator is applied to the velocity anomaly,
# producing the elevation anomaly, then the elevation anomaly is localized
# to remove off-lake deformation (in practice these are interpreted as regional trends
# but a small component may be due to the lake activity), and some noise is added
# to the synthetic elevation field

import sys
sys.path.insert(0, '../source')

import numpy as np
import os

def make_data(t_cycle,beta,eta,H):

    # make a directory for the data
    if os.path.isdir('../data/data_synth')==False:
        os.mkdir('../data/data_synth')

    # define the scalar parameters in the problem
    # in practice we might use means of some of these fields around the lake
    H = np.array([H])          # ice thickness over the lake
    beta = np.array([beta])       # basal drag coeff. near the lake
    eta = np.array([eta])        # viscosity near the lake

    # define coordinate arrays
    N_cycles = 10
    t0 = np.linspace(0,t_cycle*N_cycles,100*N_cycles)                     # time
    x0 = np.linspace(-20,20,101)*1000/H.mean()     # x coordinate
    y0 = np.linspace(-20,20,101)*1000/H.mean()     # y coordinate

    # save everything so far in the data directory
    np.save('../data/data_synth/t.npy',t0)
    np.save('../data/data_synth/x.npy',x0)
    np.save('../data/data_synth/y.npy',y0)
    np.save('../data/data_synth/x_d.npy',x0*H.mean()/1e3)
    np.save('../data/data_synth/y_d.npy',y0*H.mean()/1e3)
    np.save('../data/data_synth/eta.npy',eta)
    np.save('../data/data_synth/beta.npy',beta)
    np.save('../data/data_synth/H.npy',H)
    np.save('../data/data_synth/u.npy',np.array([0.0]))
    np.save('../data/data_synth/v.npy',np.array([0]))
    np.save('../data/data_synth/t_cycle.npy',np.array([t_cycle]))


    # import some functions to produce the synthetic elevation anomaly
    from params import t,x,y,Nt,Nx,Ny
    from operators import fwd
    from localization import localize
    from kernel_fcns import ifftd

    # set basal vertical velocity anomaly to an oscillating gaussian
    sigma = (10*1000.0/H)/3.0
    w_true = -5*np.exp(-0.5*(sigma**(-2))*(x**2+y**2))*np.cos(2*np.pi*t/t_cycle)

    # produce synthetic elevation anomaly by applying the forward operator
    # (returns fft of elevation), inverse fourier-transforming the results,
    # and removing the off-lake component ("localize" function)
    h = ifftd(fwd(w_true)).real

    # add some noise
    noise_h = 0*np.random.normal(size=(Nt,Nx,Ny),scale=np.sqrt(1e-3))
    h_obs = h + noise_h
    h_loc = localize(h_obs)
    off_lake = h_obs - h_loc
    off_lake = off_lake[:,0,0]

    np.save('../data/data_synth/h_true.npy',h)
    np.save('../data/data_synth/h_obs.npy',h_loc)
    np.save('../data/data_synth/off_lake.npy',off_lake)
    np.save('../data/data_synth/w_true.npy',w_true)
