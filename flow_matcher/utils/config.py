import os
from collections.abc import Mapping
import importlib
import pickle

def import_class(_class):
    if type(_class) is not str: return _class
    ## 'diffusion' on standard installs
    repo_name = __name__.split('.')[0]
    ## eg, 'utils'
    module_name = '.'.join(_class.split('.')[:-1])
    ## eg, 'Renderer'
    class_name = _class.split('.')[-1]
    ## eg, 'diffusion.utils'
    module = importlib.import_module(f'{repo_name}.{module_name}')
    ## eg, diffusion.utils.Renderer
    _class = getattr(module, class_name)
    # print(f'[ utils/config ] Imported {repo_name}.{module_name}:{class_name}')
    return _class

class Config(Mapping):

    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):
        self._class = import_class(_class)
        self._device = device
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            if not os.path.exists(savepath):
                pickle.dump(self, open(savepath, 'wb'))
                print(f'[ utils/config ] Saved config to: {savepath}\n')

            # Also save as JSON for better robustness/readability
            json_path = savepath.replace('.pkl', '.json')
            if not os.path.exists(json_path):
                try:
                    import json
                    def default_json(obj):
                        if hasattr(obj, '__name__'): return obj.__name__
                        return str(obj)
                    
                    with open(json_path, 'w') as f:
                        json.dump({
                            '_class': self._class.__name__,
                            **self._dict
                        }, f, indent=4, default=default_json)
                    print(f'[ utils/config ] Saved config to: {json_path}\n')
                except Exception:
                    pass

    def __repr__(self):
        string = f'\n[utils/config ] Config: {self._class}\n'
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f'    {key}: {val}\n'
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        if attr == '_dict' and '_dict' not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        instance = self._class(*args, **kwargs, **self._dict)
        if self._device:
            instance = instance.to(self._device)
        return instance
