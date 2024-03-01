#/usr/bin/env/python
import numpy as np
import camb
import sys
import time

krange1 = np.logspace(-5, -4, num=20, endpoint=False)
krange2 = np.logspace(-4, -3, num=30, endpoint=False)
krange3 = np.logspace(-3, -2, num=50, endpoint=False)
krange4 = np.logspace(-2, 0, num=200, endpoint=False)
krange5 = np.logspace(0, np.log10(20), num=50, endpoint=False)
k = np.concatenate([krange1,krange2,krange3,krange4,krange5])


def main(param_file, out_file_pk_nonu, out_file_pk_tot, out_file_sigma8, start_idx=0, end_idx=None):
    np.savetxt('k_modes.txt', k)
    params = np.genfromtxt(param_file, names=True)
    if end_idx is None:
        end_idx = len(params)
    for i in range(start_idx, end_idx):
        spectra_generation(params[i], out_file_pk_nonu, out_file_pk_tot, out_file_sigma8)


def spectra_generation(params, out_file_pk_nonu, out_file_pk_tot, out_file_sigma8):
    print(params, flush=True)
    ombh2 = params['Omegab']*params['h']**2
    summnu = params['mnu']
    omnuh2 = summnu * (3.044/3.)**0.75 / 94.06410581217612# * (TCMB/2.7255)**3
    ommh2 = params['Omegam']*params['h']**2
    omch2 = ommh2 - ombh2 - omnuh2
    t = [time.time(),]
    cp = camb.set_params(ombh2 = ombh2,
                         omch2 = omch2,
                         omk=0.0,
                         H0 = 100.*params['h'],
                         ns = 1.,
                         As = 1e-9,
                         w=params['w'],
                         WantTransfer=True,
                         kmax=20./params['h'],
                         num_massive_neutrinos=1,
                         mnu=summnu,
                         tau=0.05,
                         redshifts=[params['z'], 0.],
                         accurate_massive_neutrino_transfers=True,
                         verbose=False,)
    cp.set_accuracy(AccuracyBoost=2.,)
    # t.append(time.time())

    try:
        # Run CAMB
        results = camb.get_results(cp)
        t.append(time.time())
        # Power spectrum for cmb + baryons (no neutrinos)
        PKcamb = results.get_matter_power_interpolator(var1='delta_nonu',
                                                       var2='delta_nonu',
                                                       log_interp=True,
                                                       nonlinear=False,
                                                       hubble_units=True,
                                                       k_hunit=True,)
        Plin = PKcamb.P(z=params['z'], kh=k)
        out_array = np.append(params.tolist(), np.asarray(Plin))
        with open(out_file_pk_nonu, 'ab') as f:
            np.savetxt(f, [out_array])
        # Total matter power spectrum (cmb+baryons+neutrinos)
        PKcamb = results.get_matter_power_interpolator(var1='delta_tot',
                                                       var2='delta_tot',
                                                       log_interp=True,
                                                       nonlinear=False,
                                                       hubble_units=True,
                                                       k_hunit=True)
        for z in [0., params['z']]:
            Plin = PKcamb.P(z=z, kh=k)
            p = params.copy()
            p[-1] = z
            out_array = np.append(p.tolist(), np.asarray(Plin))
            with open(out_file_pk_tot, 'ab') as f:
                np.savetxt(f, [out_array])
        sigma8 = results.get_sigma8()[-1]
        out_array = np.append(params.tolist(), sigma8)
        with open (out_file_sigma8, 'ab') as f:
            np.savetxt(f, [out_array])
        t.append(time.time())
        print(np.diff(t), 's', flush=True)
    except:
        print('something wrong with CAMB')

    return



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    print("done")
