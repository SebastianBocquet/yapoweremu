yapoweremu: yet another emulator for the linear matter power spectrum
=====================================================================

Fast and accurate predictions for the linear matter power spectrum over a wide range of cosmological parameters, including the dark energy equation of state parameter and the sum of neutrino masses.
The emulator can provide the power spectrum for all matter (`tot`, useful for computing quantities like σ<sub>8</sub>), or the power spectrum from cold dark matter + baryons (but not neutrinos, `nonu`, useful for computing the halo mass function).

The emulator is based on a neural network implemented using pytorch, and is trained on power spectra computed using CAMB.

Why do we need yet another emulator for the linear power spectrum? I found that no existing code provides the `nonu` power spectrum over a sufficiently wide range of parameters.

Installation
------------

`yapoweremu` requires a pretty basic python installation with numpy and pytorch.

You can easily install it with `pip install https://github.com/SebastianBocquet/yapoweremu`

Useage
------

```
import poweremu

emu = poweremu.Emulator()
k = emu.k

params = {'Omegam': .3, 'Omegab': .06, 'mnu': .1, 'h': .7, 'w': -1., 'z': 0.}
pk_tot = emu.predict(param_dict=params, kind='tot')
pk_nonu = emu.predict(param_dict=params, kind='nonu')
```
The output power spectra will have the same shape as `emu.k`.
You can also provide multiple sets of parameters, such as
```
params = params = {'Omegam': [.3, .4], 'Omegab': [.06, .05], 'mnu': [0., .1], 'h': [.7, .8], 'w': [-1., -1.1], 'z': [0., 1.]}
pk_nonu = emu.predict(param_dict=params, kind='nonu')
```
In this case, the output power spectrum has shape (2, len(emu.k)).
Higher-dimensional input arrays also work, and will result in corresponding high-dimensional outputs.
