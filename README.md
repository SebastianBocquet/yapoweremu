# yapoweremu: yet another emulator for the linear matter power spectrum

Fast and accurate predictions for the linear matter power spectrum over a wide range of cosmological parameters, including the dark energy equation of state parameter and the sum of neutrino masses.
The emulator can provide the power spectrum for all matter (`tot`, useful for computing quantities like σ<sub>8</sub>), or the power spectrum from cold dark matter + baryons (but not neutrinos, `nonu`, useful for computing the halo mass function).

The emulator is based on a neural network implemented using pytorch, and is trained on power spectra computed using CAMB.

(Why) Do we need yet another emulator for the linear power spectrum? I found that no existing code provides the `nonu` power spectrum over a sufficiently wide range of parameters.

## Installation

`yapoweremu` requires a pretty basic python installation with numpy and pytorch.

You can easily install it with `pip install https://github.com/SebastianBocquet/yapoweremu`

## Tutorial

```
import poweremu

emu = poweremu.Emulator()
k = emu.k

params = {'Omegam': .3, 'Omegab': .06, 'mnu': .1, 'h': .7, 'w': -1., 'z': 0., 'ln1e10As': 3., 'ns': .97}
pk_tot = emu.predict(param_dict=params, kind='tot')
pk_nonu = emu.predict(param_dict=params, kind='nonu')
```
The output power spectra will have the same shape as `emu.k`.
You can also provide multiple sets of parameters, such as
```
params = params = {'Omegam': [.3, .4], 'Omegab': [.06, .05], 'mnu': [0., .1], 'h': [.7, .8], 'w': [-1., -1.1], 'z': [0., 1.], 'ln1e10As': [3., 3.3], 'ns': [.96, 1.]}
pk_nonu = emu.predict(param_dict=params, kind='nonu')
```
In this case, the output power spectrum has shape (2, len(emu.k)).
Higher-dimensional input arrays also work, and will result in corresponding high-dimensional outputs.

### Input parameters
The underlying neural net is only trained on the parameters `Omegam`, `Omegab`, `mnu`, `h`, `w`, `z`. These parameters need to be within the ranges

| Parameter | min  | max   |
| --------- | ---- | ----- |
| Omegam    | 0.12 | 0.5   |
| Omegab    | 0.03 | 0.07  |
| mnu       | 0.0  | 0.5   |
| h         | 0.55 | 0.85  |
| w         | -2.0 | -0.33 |
| z         | 0.0  | 3.0   |

The normalization `ln1e10As` and slope `ns` of the power spectrum can be provided within arbitrary ranges

| Parameter | min | max |
| --------- | --- | --- |
| ln1e10As  | -   | -   |
| ns        | -   | -   |

*Note* The neural net is actually trained for 0.1 < Omegam < 0.5. However, for combinations of Omegam<0.12 and large Omegab (around 0.07), the emulator does not perform well (because baryonic features in the power spectrum become extremely pronounced. The user can override the default `Omegam>0.12` when initializing the emulator as `emu = poweremu.Emulator(live_dangerously=True)`, this will set the lower limit to 0.1.
