import numpy as np
import torch
import pkg_resources

class Emulator:
    def __init__(self, live_dangerously=False):
        """Load all data and setup the neural network.

        :param live_dangerously: If True, the lower limit of `Omegam` is set to 0.12.
        This results in better performance, because in this case, there is at least
        as much cold dark matter as there is baryonic matter.
        :type live_dangerously: bool
        """
        f_name = pkg_resources.resource_filename(__name__, 'param_ranges.txt')
        self.param_ranges = np.genfromtxt(f_name, names=True)
        if not live_dangerously:
            self.param_ranges['Omegam'][0] = 0.12
        self.param_names_nn = self.param_ranges.dtype.names
        self.param_names = ['Omegam', 'Omegab', 'mnu', 'h', 'w', 'z', 'ln1e10As', 'ns']
        self.k = np.loadtxt(pkg_resources.resource_filename(__name__, 'k.txt'))
        self.emu = {}
        for kind in ['nonu', 'tot']:
            tmp = np.load(pkg_resources.resource_filename(__name__, 'regularize_%s.npy'%kind))
            self.emu[kind] = {'means':tmp[0], 'stds':tmp[1]}
            f_name = pkg_resources.resource_filename(__name__, 'pk_%s.pt'%kind)
            self.emu[kind]['model'] = torch.jit.load(f_name)
            self.emu[kind]['model'].eval()

    def predict(self,
                kind=None,
                param_dict=None):
        """Predict the power spectrum for a given set of parameters. The
        wavenumber `Emulator.k` is in units of h/Mpc, the power spectrum is in
        units of (h^{-1}Mpc)^3.

        :param kind: Power spectrum for cold dark matter and baryons (`nonu`) or
        for all matter (including neutrinos, `tot`).
        :type kind: str
        :param param_dict: Dictionary of parameters. Must contain all parameters
        :type param_dict: dict

        :return: Natural logarithm of matter power spectrum in units of (h^{-1}Mpc)^3.
        Shape is (param_dict[name].shape, len(k))
        :type return: np.ndarray
	"""
        # Validate input
        assert kind in ['nonu', 'tot'], "kind must be either 'nonu' or 'tot'"
        self.validate_params(param_dict)
        # Create torch tensor
        x_np = np.array([param_dict[name] for name in self.param_ranges.d]).T
        x = torch.tensor(x_np).float()
        # Evaluate model
        with torch.no_grad():
            pred = self.emu[kind]['model'](x).numpy() * self.emu[kind]['stds']
        pred+= self.emu[kind]['means']
        return pred

    def validate_params(self, param_dict):
        """Check `param_dict` for most likely errors"""
        # Check if all parameters are present
        for name in self.param_names: 
            assert name in param_dict, "param_dict must contain %s"%name
        # Check if the parameters that enter the NN are within the allowed range
        for name in self.param_names_nn:
            assert np.all((param_dict[name] >= self.param_ranges[name][0]) & (param_dict[name] <= self.param_ranges[name][1])), "param_dict[%s] must be within the range %s"%(name, self.param_ranges[name])
        # Check if all parameters have the same shape
        all_shape = [np.shape(param_dict[name]) for name in self.param_names]
        assert len(set(all_shape)) == 1, "All parameters must have the same shape"
        return 0
