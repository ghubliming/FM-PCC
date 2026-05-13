import os
import importlib
import importlib.util
import shutil
import random
import numpy as np
import torch
import json
# from tap import 
import argparse
import pdb

from .serialization import mkdir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(argparse.ArgumentParser):

    def __init__(self, *args, savepath='', **kwargs):
        # Remove custom arguments before passing to super().__init__
        self.exe_name = kwargs.pop('exe_name', None)
        self.dataset = kwargs.pop('dataset', None)
        self.config = kwargs.pop('config', None)
        super().__init__(*args, **kwargs)
        self.add_argument('--config', help='Path to config file')
        self.add_argument('--seed', type=int, help='Random seed')  # Add seed argument
        self.savepath = savepath

    def save(self, args):
        fullpath = os.path.join(self.savepath, 'args.json')
        
        # If primary args.json already exists, we are resuming/re-running.
        # Preserve the original and save the current as a numbered version.
        if os.path.exists(fullpath):
            json_base = fullpath.replace('.json', '')
            resume_idx = 1
            while os.path.exists(f'{json_base}_resume_{resume_idx}.json'):
                resume_idx += 1
            fullpath = f'{json_base}_resume_{resume_idx}.json'

        with open(fullpath, 'w') as f:
            json.dump(vars(args), f, skipkeys=True)
        # print(f'[ utils/setup ] Saved args to {fullpath}')

    def parse_args(self, experiment=None, seed=None):
        args = super().parse_args()
        args.dataset = self.dataset
        args.config = self.config
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment)

        args.seed = seed if seed is not None else args.seed

        # self.add_extras(args)
        self.eval_fstrings(args)
        self.set_seed(args)
        self.set_loadbase(args)
        self.generate_exp_name(args)
        
        # Only save args if we are training. In evaluation, we want to avoid 
        # creating redundant and confusing 'args_resume_X.json' files.
        save = (experiment == 'train')
        self.mkdir(args, save=save)
        return args

    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        # print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            # print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        # else:
            # print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        # print(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            # print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                # print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new
                setattr(args, key, new)     # Added so that it works with argparse

    def set_seed(self, args):
        if not hasattr(args, 'seed') or args.seed is None:
            return
        # print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, 'loadbase') and args.loadbase is None:
            # print(f'[ utils/setup ] Setting loadbase: {args.logbase}')
            args.loadbase = args.logbase

    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            # print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args, save=True):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name, str(args.seed))
            self.savepath = args.savepath
            self._dict['savepath'] = args.savepath
            # if 'suffix' in dir(args):
            #     args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')

            # Smart Config Snapshot
            self.snapshot_configs(args)
            if save:
                self.save(args)
    def snapshot_configs(self, args):
        if not hasattr(args, 'config'):
            return

        # Create subfolder named after the config module
        config_name = args.config.split('.')[-1]
        snapshot_dir = os.path.join(args.savepath, f'config_snapshot_{config_name}')
        
        os.makedirs(snapshot_dir, exist_ok=True)

        # 1. Copy the main python config module file
        try:
            import importlib.util
            import shutil
            spec = importlib.util.find_spec(args.config)
            if spec and spec.origin:
                dest = os.path.join(snapshot_dir, os.path.basename(spec.origin))
                shutil.copy(spec.origin, dest)
                # print(f'[ utils/setup ] Snapshotted config to {dest}')
        except Exception:
            pass

        # 2. Copy associated yaml configs (e.g. projection_eval.yaml)
        # We look in the 'config/' directory relative to the current working directory
        yaml_path = 'config/projection_eval.yaml'
        if os.path.exists(yaml_path):
            try:
                dest = os.path.join(snapshot_dir, 'projection_eval.yaml')
                shutil.copy(yaml_path, dest)
                # print(f'[ utils/setup ] Snapshotted config to {dest}')
            except Exception:
                pass

        # 3. Create a timestamp file AFTER copying is finished
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(snapshot_dir, f'snapshot_{timestamp}'), 'w') as f:
                f.write(f'Snapshot taken at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        except Exception:
            pass
