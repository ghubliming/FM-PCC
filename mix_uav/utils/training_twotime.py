import os, pickle
import copy
import math
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup

from .arrays import batch_to_device

# ── Gen3v7 — extra per-step metrics from AlphaFlowODE._build_info ─────────────────────
# Tracked generically (train + test) and persisted as training_<key>_losses /
# test_<key>_losses in losses.pkl / losses.json. `h_mse_b*` is the h-stratified residual
# inherited from Gen3v6 — the metric COMPARE §7.4.1 asked for and no generation had.
# Empty h-buckets emit NaN from the model; NaN points are DROPPED here, never logged as 0.
#
# ⭐ Gen3v7 adds the SCHEDULE TELEMETRY: `alpha`, `discrete_frac`, `clamp_frac`. Gate G4 is
# exactly "α is logged and moving, and the branch counts shift over training" — a run whose
# α never left 1.0 trained plain flow matching and is otherwise indistinguishable from a
# working one (PLAN §11 trap 1).
EXTRA_METRIC_KEYS = (
    'raw_mse_u', 'raw_mse_v', 'per_dim_rms_u',
    'h_mse_b0', 'h_mse_b1', 'h_mse_b2', 'h_mse_b3',
    'h_mean', 'fm_frac',
    'alpha', 'discrete_frac', 'clamp_frac',
)


def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        train_test_split=1.0,
        split_seed=42,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        lr_warmup_steps=1000,
        n_train_steps=1e5,
        n_steps_per_epoch=1000,
        gradient_accumulate_every=2,
        gradient_clip=0.0,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        train_device='cuda',
        results_folder='./results',
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.save_freq = n_train_steps // 5

        self.n_train_steps = n_train_steps
        self.n_steps_per_epoch = n_steps_per_epoch
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # 🔴 Gen3v6/Gen3v7 — `gradient_clip` was a DEAD config key in this lineage: it existed in
        # config/avoiding-d3il.py and NO trainer ever read it (POST_U10_III §4.1), while both
        # Gen3v4 and Gen13 logged 65–500× loss spikes. Here it is actually applied, in
        # train_epoch(), right before optimizer.step(). 0 (or negative) disables it.
        self.gradient_clip = float(gradient_clip)

        self.dataset = dataset
        self.train_test_split = train_test_split
        if train_test_split == 1:
            self.train_dataloader = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
            ))
        else:
            n_train = int(train_test_split * len(self.dataset))
            n_test = len(self.dataset) - n_train
            # U9: seeded generator — unseeded split re-splits differently on resume,
            # leaking old test trajectories into training and making best_test_loss
            # compare across different test sets.
            train_dataset, test_dataset = torch.utils.data.random_split(
                self.dataset, [n_train, n_test],
                generator=torch.Generator().manual_seed(split_seed),
            )
            self.train_dataloader = cycle(torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
            ))
            self.test_dataloader = cycle(torch.utils.data.DataLoader(
                test_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
            ))
            self.best_test_loss = np.inf
        self.train_losses = []
        self.test_losses = []
        self.train_a0_losses = []
        self.test_a0_losses = []
        self.test_raw_mse_losses = []  # U9: adaptive-weight-free companion val metric
        # U9 metric-parity pass: train-side raw MSE (the reference repos' loss_u analog) and
        # the v-head loss on both sides (their loss_v analog). In Gen3v7 these are aliases of
        # raw_mse_u / raw_mse_v — kept so the inherited pkl keys and DA tooling still work.
        self.train_raw_mse_losses = []
        self.train_aux_losses = []
        self.test_aux_losses = []
        # Gen3v7: h-stratified + raw-MSE metric family (see EXTRA_METRIC_KEYS)
        self.train_extra_losses = {k: [] for k in EXTRA_METRIC_KEYS}
        self.test_extra_losses = {k: [] for k in EXTRA_METRIC_KEYS}
        # Gen3v7: observed grad-norm BEFORE clipping — tells you whether gradient_clip is
        # actually biting, and is the direct diagnostic for the inherited loss-spike problem.
        self.grad_norm_history = []
        # U9 parity: learning-rate curve — verifies the cosine+warmup scheduler
        # resumed correctly (it is rebuilt from scratch on resume) and explains
        # loss spikes/plateaus at a glance
        self.lr_history = []
        self.current_test_loss = None
        self.current_test_a0_loss = None

        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=self.n_train_steps,
        )

        self.logdir = results_folder

        self.reset_parameters()
        self.step = 0

        self.device = train_device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train_epoch(self, n_train_steps, epoch=0):        
        progress_bar = tqdm(total=n_train_steps, mininterval=1e10)
        progress_bar.set_description(f"Epoch {epoch}")

        for step in range(n_train_steps):
            # 🔴 Gen3v7 (PLAN §3.7) — push the GLOBAL OPTIMIZER step into the model so the
            # α schedule can read it. `self.step` is incremented once per optimizer step;
            # the inner `for step in range(...)` above is the ACCUMULATION loop and using it
            # would halve the effective schedule length. Placed outside the accumulation
            # loop so every micro-batch of one optimizer step sees the same α.
            # Resume-safe: load() restores self.step before training continues, so α never
            # restarts at 1.0 on a requeue (PLAN §11 trap 6).
            if hasattr(self.model, 'set_train_step'):
                self.model.set_train_step(self.step)
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            # Gen3v6/Gen3v7: gradient clipping is APPLIED here (it was a dead key upstream).
            # clip_grad_norm_ returns the pre-clip total norm, which we log.
            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip)
            else:
                grad_norm = None

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step
                self.save(label)

            if self.step % self.log_freq == 0:
                self.train_losses.append([self.step, loss.item()])
                if 'a0_loss' in infos:
                    self.train_a0_losses.append([self.step, infos['a0_loss'].item()])
                if 'raw_mse' in infos:
                    self.train_raw_mse_losses.append([self.step, infos['raw_mse'].item()])
                if 'aux_loss' in infos:
                    self.train_aux_losses.append([self.step, infos['aux_loss'].item()])
                for key in EXTRA_METRIC_KEYS:
                    if key in infos:
                        value = infos[key].item()
                        if not math.isnan(value):   # empty h-bucket ⇒ skip, do not log 0
                            self.train_extra_losses[key].append([self.step, value])
                if grad_norm is not None:
                    self.grad_norm_history.append([self.step, float(grad_norm)])
                self.lr_history.append([self.step, self.lr_scheduler.get_last_lr()[0]])

                if self.train_test_split < 1:
                    test_loss, test_a0_loss, test_raw_mse, test_aux, test_extra = self.test()
                    self.test_losses.append([self.step, test_loss])
                    self.test_a0_losses.append([self.step, test_a0_loss])
                    self.test_raw_mse_losses.append([self.step, test_raw_mse])
                    if test_aux is not None:
                        self.test_aux_losses.append([self.step, test_aux])
                    for key, value in test_extra.items():
                        self.test_extra_losses[key].append([self.step, value])
                    self.current_test_loss = test_loss
                    self.current_test_a0_loss = test_a0_loss
                    if test_loss < self.best_test_loss:
                        self.best_test_loss = test_loss
                        self.save_best()
                    if self.step == 0:
                        # if self.model.action_dim > 0:
                        print(f'Initial test loss: {test_loss:8.4f}, a0 loss: {test_a0_loss:8.4f}')
                        # else:
                            # print(f'Initial test loss: {test_loss:8.4f}')
                self.save_losses()
            
            # if self.train_test_split < 1:
            #     if 'a0_loss' in infos:
            #         logs = {"loss": loss.item(), "a0_loss": infos['a0_loss'].item(), "loss_test": test_loss, "a0_loss_test": test_a0_loss, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}
            #     else:
            #         logs = {"loss": loss.item(), "loss_test": test_loss, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}
            # else:
            #     if 'a0_loss' in infos:
            #         logs = {"loss": loss.item(), "a0_loss": infos['a0_loss'].item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}
            #     else:
            #         logs = {"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}

            logs = {"loss": loss.item()}
            for l in ["diffusion_loss", "dyn_loss", "a0_loss", "raw_mse", "aux_loss",
                      "raw_mse_u", "per_dim_rms_u",
                      "alpha", "discrete_frac"]:     # Gen3v7 schedule telemetry
                if l in infos:
                    logs[l] = infos[l].item()
            if self.train_test_split < 1 and self.current_test_loss is not None:
                logs["loss_test"] = self.current_test_loss
                if 'a0_loss' in infos:
                    logs["a0_loss_test"] = self.current_test_a0_loss
            logs["lr"] = self.lr_scheduler.get_last_lr()[0]
            logs["step"] = self.step

            if (self.step + 1) % self.log_freq == 0 or step == n_train_steps - 1:
                progress_bar.update(step - progress_bar.n + 1)
                progress_bar.set_postfix(**logs)

            self.step += 1

    def train(self, on_epoch_end=None):
        if self.step >= self.n_train_steps:
            return

        remaining_steps = int(self.n_train_steps - self.step)
        epoch = int(self.step // self.n_steps_per_epoch)

        while remaining_steps > 0:
            steps_this_epoch = min(self.n_steps_per_epoch, remaining_steps)
            self.train_epoch(steps_this_epoch, epoch)
            remaining_steps -= steps_this_epoch
            epoch += 1
            # U9: lets callers flush losses to W&B per epoch instead of only at the end
            if on_epoch_end is not None:
                on_epoch_end(epoch)

        # ── Gen15 U6 ── SAVE THE MODEL THE SCHEDULE ACTUALLY ENDS ON.
        #
        # 🔴 The periodic save fires on `self.step % self.save_freq == 0` inside train_epoch,
        # and self.step only ever reaches n_train_steps - 1 there. So the newest NUMERIC
        # checkpoint is the last multiple of save_freq strictly BELOW n_train_steps — at the
        # default save_freq = n_train_steps // 5 that is step 80000 of 100000. `--epoch latest`
        # has therefore always deployed a model 20% short of the end of training, and no save
        # cadence fixes it: the last multiple of ANY frequency is < n_train_steps. The eval's
        # own --epoch help text has documented this as a known wart; U6 removes the wart.
        #
        # Fires ONLY on a completed run: the early return at the top of this method means
        # there are steps left to do and that run's periodic saves still stand. Costs one extra
        # state_<n_train_steps>.pt per completed run.
        if self.step >= self.n_train_steps:
            _last_periodic = (int(self.n_train_steps) - 1) // self.save_freq * self.save_freq
            print(f'[ utils/training_twotime ] final checkpoint: step {self.step} '
                  f'(the periodic save only reaches {_last_periodic}); '
                  f'"latest" now resolves to the end of the schedule', flush=True)
            self.save(self.step)

    def test(self, n_test=100):
        self.model.eval()   # Set the model to evaluation mode

        test_loss = 0
        test_a0_loss = 0
        test_raw_mse = 0
        test_aux = 0
        aux_seen = False
        # Gen3v7: NaN-safe running means for the extra metric family (empty h-buckets)
        extra_sum = {k: 0.0 for k in EXTRA_METRIC_KEYS}
        extra_cnt = {k: 0 for k in EXTRA_METRIC_KEYS}
        with torch.no_grad():
            for step in range(n_test):
                batch = next(self.test_dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss /= self.gradient_accumulate_every

                test_loss += loss.item()
                test_a0_loss += infos['a0_loss'].item() if 'a0_loss' in infos else 0
                test_raw_mse += infos['raw_mse'].item() if 'raw_mse' in infos else 0
                if 'aux_loss' in infos:
                    test_aux += infos['aux_loss'].item()
                    aux_seen = True
                for key in EXTRA_METRIC_KEYS:
                    if key in infos:
                        value = infos[key].item()
                        if not math.isnan(value):
                            extra_sum[key] += value
                            extra_cnt[key] += 1

            test_loss /= n_test
            test_a0_loss /= n_test
            test_raw_mse /= n_test
            # None (not 0) when the objective has no aux head — callers skip logging it
            test_aux = test_aux / n_test if aux_seen else None
            test_extra = {k: extra_sum[k] / extra_cnt[k]
                          for k in EXTRA_METRIC_KEYS if extra_cnt[k] > 0}

        # U9: restore train mode — eval() above was never undone, leaving the model
        # in eval mode for all training after step 0 (inherited from DPCC upstream).
        # No effect on current backbones (mask-based condition dropout only), but any
        # future nn.Dropout/BatchNorm layer would silently train wrong without this.
        self.model.train()

        return test_loss, test_a0_loss, test_raw_mse, test_aux, test_extra

    def _checkpoint_payload(self):
        '''common state for save()/save_best() — one place so the two never drift'''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            # 🔴 fix_6 — OPTIMIZER STATE. Without these two, "resume" restarted Adam with
            # zeroed moments and rewound the cosine LR schedule to its warmup, i.e. at step
            # 80000 the LR jumped 4.87e-5 -> 3e-4. That is not a resume, it is a second
            # training run glued onto a checkpoint. Old checkpoints lack these keys; load()
            # degrades gracefully (see there).
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_test_loss': self.best_test_loss if hasattr(self, 'best_test_loss') else None,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_a0_losses': self.train_a0_losses,
            'test_a0_losses': self.test_a0_losses,
            'test_raw_mse_losses': self.test_raw_mse_losses,
            'train_raw_mse_losses': self.train_raw_mse_losses,
            'train_aux_losses': self.train_aux_losses,
            'test_aux_losses': self.test_aux_losses,
            'lr_history': self.lr_history,
            'grad_norm_history': self.grad_norm_history,   # Gen3v7
        }
        for key in EXTRA_METRIC_KEYS:                      # Gen3v7
            data[f'train_{key}_losses'] = self.train_extra_losses[key]
            data[f'test_{key}_losses'] = self.test_extra_losses[key]
        return data

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(self._checkpoint_payload(), savepath)
        # print(f'Saved model to {savepath}', flush=True)

    def save_best(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        savepath = os.path.join(self.logdir, f'state_best.pt')
        torch.save(self._checkpoint_payload(), savepath)
        # print(f'Saved best model to {savepath}', flush=True)

    def save_losses(self):
        current_losses = {
            'training_losses': self.train_losses,
            'test_losses': self.test_losses,
            'training_a0_losses': self.train_a0_losses,
            'test_a0_losses': self.test_a0_losses,
            'test_raw_mse_losses': self.test_raw_mse_losses,
            'training_raw_mse_losses': self.train_raw_mse_losses,
            'training_aux_losses': self.train_aux_losses,
            'test_aux_losses': self.test_aux_losses,
            'lr_history': self.lr_history,
            'grad_norm_history': self.grad_norm_history,   # Gen3v7
        }
        # Gen3v7 extra metric family (h-stratified residuals, raw MSEs, sampler sanity)
        for key in EXTRA_METRIC_KEYS:
            current_losses[f'training_{key}_losses'] = self.train_extra_losses[key]
            current_losses[f'test_{key}_losses'] = self.test_extra_losses[key]
        savepath = os.path.join(self.logdir, 'losses.pkl')

        # If existing losses exist on disk, merge them to avoid overwriting on resume
        if os.path.isfile(savepath):
            try:
                with open(savepath, 'rb') as f:
                    data = pickle.load(f)
                
                for key in current_losses:
                    if key in data:
                        # Use dict for merging to prefer NEW entries for any overlapping steps
                        # This handles "rewind" resumes where step history is being redone
                        merged_dict = {entry[0]: entry for entry in data[key]}
                        for entry in current_losses[key]:
                            merged_dict[entry[0]] = entry
                        data[key] = [merged_dict[s] for s in sorted(merged_dict.keys())]
                    else:
                        data[key] = current_losses[key]
            except Exception as e:
                print(f'[ utils/training ] Error merging losses at {savepath}: {e}')
                data = current_losses
        else:
            data = current_losses

        with open(savepath, 'wb') as f:
            pickle.dump(data, f)

        # 2. Update the .json exhaustive log ("save all we done")
        # The JSON version will be a cumulative history of all reported points, chronologically.
        json_path = savepath.replace('.pkl', '.json')
        try:
            import json
            history = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    history = json.load(f)
            
            for key in current_losses:
                if key not in history:
                    history[key] = []
                # Append only truly NEW points from this session to the history
                # (Assuming Trainer.train_losses only grows during a session)
                existing_steps = {tuple(e) for e in history[key]}
                new_entries = [e for e in current_losses[key] if tuple(e) not in existing_steps]
                history[key].extend(new_entries)
            
            with open(json_path, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f'[ utils/training ] Error saving exhaustive losses to {json_path}: {e}')

    def _restore_optimizer_state(self, data):
        '''
            fix_6 — put the optimizer and the LR schedule back where they were.

            Two cases:
              * checkpoint written by fix_6 or later -> exact restore.
              * checkpoint written BEFORE fix_6 (no 'optimizer'/'lr_scheduler' keys) -> the
                Adam moments are unrecoverable, but the LR schedule is a pure function of
                the step count, so we replay it forward to self.step. That alone is the
                difference between continuing at 4.87e-5 and restarting warmup at 3e-4.
            Call AFTER self.step has been set.

            Gen3v7 note: the α schedule needs no restoring — `train_epoch` recomputes it
            from `self.step` via `set_train_step` on every step, so a resumed run picks the
            anneal up exactly where it left off. Check train/alpha after a resume anyway.
        '''
        if 'best_test_loss' in data and data['best_test_loss'] is not None:
            self.best_test_loss = data['best_test_loss']

        if 'optimizer' in data and 'lr_scheduler' in data:
            self.optimizer.load_state_dict(data['optimizer'])
            self.lr_scheduler.load_state_dict(data['lr_scheduler'])
            print(f'[ utils/training ] Restored optimizer + LR schedule '
                  f'(lr={self.lr_scheduler.get_last_lr()[0]:.3e})')
            return

        print('[ utils/training ] ⚠️  Pre-fix_6 checkpoint: no optimizer state stored. '
              'Adam moments restart cold; fast-forwarding the LR schedule instead.')
        for _ in range(int(self.step)):
            self.lr_scheduler.step()
        print(f'[ utils/training ] LR schedule fast-forwarded to step {self.step} '
              f'(lr={self.lr_scheduler.get_last_lr()[0]:.3e})')

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self._restore_optimizer_state(data)

        # Restore losses from checkpoint if available
        if 'train_losses' in data:
            self.train_losses = data['train_losses']
            self.test_losses = data.get('test_losses', [])
            self.train_a0_losses = data.get('train_a0_losses', [])
            self.test_a0_losses = data.get('test_a0_losses', [])
            self.test_raw_mse_losses = data.get('test_raw_mse_losses', [])  # U9
            self.train_raw_mse_losses = data.get('train_raw_mse_losses', [])  # U9 parity
            self.train_aux_losses = data.get('train_aux_losses', [])          # U9 parity
            self.test_aux_losses = data.get('test_aux_losses', [])            # U9 parity
            self.lr_history = data.get('lr_history', [])                      # U9 parity
            self.grad_norm_history = data.get('grad_norm_history', [])        # Gen3v7
            for key in EXTRA_METRIC_KEYS:                                     # Gen3v7
                self.train_extra_losses[key] = data.get(f'train_{key}_losses', [])
                self.test_extra_losses[key] = data.get(f'test_{key}_losses', [])
            print(f'[ utils/training ] Restored loss history from checkpoint at step {self.step}')
        else:
            # Fallback to losses.pkl if not in checkpoint
            losses_path = os.path.join(self.logdir, 'losses.pkl')
            if os.path.exists(losses_path):
                try:
                    with open(losses_path, 'rb') as f:
                        losses = pickle.load(f)
                    self.train_losses = losses.get('training_losses', [])
                    self.test_losses = losses.get('test_losses', [])
                    self.train_a0_losses = losses.get('training_a0_losses', [])
                    self.test_a0_losses = losses.get('test_a0_losses', [])
                    self.test_raw_mse_losses = losses.get('test_raw_mse_losses', [])  # U9
                    self.train_raw_mse_losses = losses.get('training_raw_mse_losses', [])  # U9 parity
                    self.train_aux_losses = losses.get('training_aux_losses', [])          # U9 parity
                    self.test_aux_losses = losses.get('test_aux_losses', [])               # U9 parity
                    self.lr_history = losses.get('lr_history', [])                         # U9 parity
                    self.grad_norm_history = losses.get('grad_norm_history', [])           # Gen3v7
                    for key in EXTRA_METRIC_KEYS:                                          # Gen3v7
                        self.train_extra_losses[key] = losses.get(f'training_{key}_losses', [])
                        self.test_extra_losses[key] = losses.get(f'test_{key}_losses', [])
                    print(f'[ utils/training ] Restored loss history from {losses_path}')
                except Exception as e:
                    print(f'[ utils/training ] Error loading losses from {losses_path}: {e}')

        # 🔴 fix_6 — recover the best-val watermark on PRE-fix_6 checkpoints.
        # Without this, best_test_loss stays np.inf after a resume, so the FIRST test
        # unconditionally calls save_best() and overwrites state_best.pt — with a model that
        # may well be worse. state_best.pt is exactly what eval loads
        # ('diffusion_epoch': 'best', config/avoiding-d3il.py:1304), so a resume would
        # silently downgrade the deliverable, and if the periodic state_*.pt files had been
        # deleted to free disk there would be nothing left to fall back to.
        # The restored test_losses history holds the exact watermark — nothing is estimated.
        if getattr(self, 'best_test_loss', None) in (None, np.inf) and len(self.test_losses) > 0:
            self.best_test_loss = min(value for _step, value in self.test_losses)
            print(f'[ utils/training ] Recovered best_test_loss={self.best_test_loss:.6f} '
                  f'from restored history ({len(self.test_losses)} points)')
