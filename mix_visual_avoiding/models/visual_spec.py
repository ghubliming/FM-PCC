"""Gen16 — THE single source of truth for the visual-AVOIDING observation spec.

WHY THIS FILE EXISTS
────────────────────────────────────────────────────────────────────────────────
Gen14 (`mix_visual_aligning/`) hardcodes the task's observation spec in NINE places:

    TRANSITION_DIM / LATENT_DIM       visual_unet.py, visual_unet_twotime.py,
                                      visual_dit_twotime.py
    'in_hand_image' in shape_meta     the same three files
    `bp, inhand, obs_seq = cond[...]` the same three + visual_gaussian_diffusion.py,
                                      visual_fm_diffusion.py, visual_mf_diffusion.py,
                                      visual_af_diffusion.py
    'wrist_img' key                   datasets/sequence.py and the four wrappers above

That was fine while there was ONE visual task. Gen16 is the second, and it differs in
exactly those nine places (2-D not 3-D, one camera not two). Replaying the same edit
nine times per new ML bone is how a bone silently keeps the other task's camera count.

So the spec is hoisted here ONCE and every consumer reads it. The rule for this
generation: **no module below `models/` may name a camera, a latent width or a
trajectory dimension.** If you are about to write `9`, `128`, `'in_hand_image'` or
`'wrist_img'` in this package, you are writing it in the wrong file.

THE AVOIDING SPEC (Gen9 Epoch 2, `fm_visual_avoiding/`)
────────────────────────────────────────────────────────────────────────────────
Trajectory layout (the 2-D analogue of aligning's 9-D):

    x[t] = [ dx   dy | des_x des_y | c_x  c_y ]
             act(2)    des_xy(2)     c_xy(2)
             idx 0-1   idx 2-3       idx 4-5

Why 6-D: the DPCC projector enforces Euler dynamics on the ACTUAL robot position
(`c_xy`, indices 4-5). The avoiding task is planar — the robot moves in a plane, the
obstacles are fixed 2-D discs, and z is held by the env, not part of the action space.

Why ONE camera: avoiding has no grasping, so the wrist/in-hand camera sees nothing the
task depends on. Only bp-cam sees the obstacle field. Gen9 Epoch 2 plan §6.

Why NO obstacle positions in obs: D3IL avoiding's `get_obj_xy_list()` returns six fixed
(x, y) pairs identical across every reset — environment constants, not state. They are
`sphere_outside` projector constraints in the planning config, never obs dims.

🔴 LATENT_DIM IS DERIVED, NOT CHOSEN. `MultiImageObsEncoder` concatenates one
`RGB_OUTPUT_SIZE`-wide feature per camera, so LATENT_DIM == N_CAMERAS *
RGB_OUTPUT_SIZE == 1 * 64 == 64 here (Gen14 aligning: 2 * 64 == 128). Setting it by
hand is how a camera count and a FiLM `cond_dim` drift apart into a shape error 40
minutes into a GPU allocation.
"""

# ── cameras ───────────────────────────────────────────────────────────────────
# Encoder keys, in the order MultiImageObsEncoder concatenates them. The dataset's
# per-camera condition keys are listed in the SAME order, so `zip()` is the only
# pairing anyone needs.
CAMERA_KEYS = ('agentview_image',)
# The condition-dict key each camera arrives under from the dataset / eval loop.
COND_IMG_KEYS = ('primary_img',)

N_CAMERAS = len(CAMERA_KEYS)
assert len(COND_IMG_KEYS) == N_CAMERAS, 'one condition key per camera'

IMG_SHAPE       = [3, 96, 96]   # must match ParityAvoidingDataset's resize
RGB_OUTPUT_SIZE = 64            # per-camera ResNet feature width
LATENT_DIM      = N_CAMERAS * RGB_OUTPUT_SIZE   # 64 — DERIVED, see the header

# ── trajectory dims ───────────────────────────────────────────────────────────
ACTION_DIM     = 2   # [dx, dy]
OBS_DIM        = 4   # [des_xy(2), c_xy(2)]
TRANSITION_DIM = ACTION_DIM + OBS_DIM   # 6
assert TRANSITION_DIM == 6

# The obs width when `if_vision=False`. Gen14 had a real state-only arm (23-D aligning
# with box/target poses); avoiding has no such sibling dataset, so a non-visual run here
# is the SAME 4-D obs with the camera removed. Kept as a named constant rather than a
# literal so `visual_unet.py`'s else-branch has one place to read.
STATE_ONLY_OBS_DIM = OBS_DIM

# Human-readable layout, printed by the backbones at construction so a batch log shows
# which task's spec is live before any tensor is allocated.
LAYOUT = ('6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)]  ·  '
          f'{N_CAMERAS} camera ({", ".join(CAMERA_KEYS)})')


# ── encoder construction ──────────────────────────────────────────────────────

def shape_meta():
    """The `shape_meta` dict MultiImageObsEncoder wants — one entry per camera."""
    return {'obs': {key: {'shape': list(IMG_SHAPE), 'type': 'rgb'} for key in CAMERA_KEYS}}


def obs_encoder_cfg():
    """The OmegaConf node for `hydra.utils.instantiate`.

    🔴 Every field except `shape_meta` is BYTE-IDENTICAL to Gen14's
    (`mix_visual_aligning/models/visual_unet.py:38-51`) and to Gen9's avoiding encoder.
    `share_rgb_model: False` is kept even at N_CAMERAS == 1, where it is inert, so the
    aligning and avoiding encoders remain the same object under a camera-count change.
    """
    from omegaconf import OmegaConf
    return OmegaConf.create({
        '_target_': 'agents.models.vision.multi_image_obs_encoder.MultiImageObsEncoder',
        'shape_meta': shape_meta(),
        'rgb_model': {
            '_target_': 'agents.models.vision.model_getter.get_resnet',
            'input_shape': list(IMG_SHAPE),
            'output_size': RGB_OUTPUT_SIZE,
        },
        'resize_shape':    None,
        'random_crop':     False,
        'use_group_norm':  True,
        'share_rgb_model': False,
        'imagenet_norm':   True,
    })


def build_obs_encoder(device):
    """Instantiate the shared vision encoder on `device`. One call site per backbone."""
    import hydra
    return hydra.utils.instantiate(obs_encoder_cfg()).to(device)


def build_obs_dict(cam_imgs):
    """(B*T, C, H, W) tensors, one per camera -> the encoder's input dict.

    `cam_imgs` must be in CAMERA_KEYS order. Arity is checked because a silently
    dropped camera produces a HALF-WIDTH latent, and the FiLM projection would then
    fail with a shape error far from its cause.
    """
    if len(cam_imgs) != N_CAMERAS:
        raise ValueError(
            f'[ visual_spec ] expected {N_CAMERAS} camera tensor(s) '
            f'{CAMERA_KEYS}, got {len(cam_imgs)}. The visual payload and this '
            f'task spec disagree — check who packed cond[\'visual\'].')
    return dict(zip(CAMERA_KEYS, cam_imgs))


# ── condition-payload packing / unpacking ─────────────────────────────────────
# The `cond['visual']` payload is (img_0, ..., img_{N-1}, obs_seq): the cameras in
# CAMERA_KEYS order followed by the obs sequence. Gen14's two-camera form
# `(bp, inhand, obs_seq)` is exactly this with N_CAMERAS == 2, so the two generations
# use the SAME convention — only the arity differs, and only this module knows it.

def pack_visual(cam_imgs, obs_seq):
    """Build the `cond['visual']` payload from per-camera tensors + the obs sequence."""
    return (*cam_imgs, obs_seq)


def split_visual(payload):
    """`cond['visual']` -> (tuple_of_camera_tensors, obs_seq_or_None).

    Tolerates a payload with no trailing obs_seq — the backbones only ever use the
    images, and Gen9's eval agent packed `(bp_imgs, obs_seq)` while some call sites
    pack images alone. Anything SHORTER than N_CAMERAS is an error, not a default.
    """
    if not isinstance(payload, (tuple, list)):
        raise TypeError(
            f'[ visual_spec ] cond["visual"] must be a tuple/list, got {type(payload)!r}.')
    if len(payload) < N_CAMERAS:
        raise ValueError(
            f'[ visual_spec ] cond["visual"] carries {len(payload)} element(s) but this '
            f'task has {N_CAMERAS} camera(s) {CAMERA_KEYS}. Refusing to run blind.')
    cam_imgs = tuple(payload[:N_CAMERAS])
    obs_seq  = payload[N_CAMERAS] if len(payload) > N_CAMERAS else None
    return cam_imgs, obs_seq


def images_from_conditions(conditions):
    """Pull the per-camera image tensors out of a DATASET condition dict.

    The dataset hands the trainer `{0: obs_anchor, 'primary_img': (B,C,H,W), ...}`.
    Each is unsqueezed to the (B, 1, C, H, W) window shape `encode_visual` expects,
    so the four engine wrappers share one line instead of four hand-written unpacks.
    A missing key raises HERE, naming the key — not later as a KeyError inside a
    JVP closure.
    """
    out = []
    for key in COND_IMG_KEYS:
        if key not in conditions:
            raise KeyError(
                f'[ visual_spec ] condition dict has no {key!r}; this task expects '
                f'{list(COND_IMG_KEYS)} (one per camera in {list(CAMERA_KEYS)}).')
        out.append(conditions[key].unsqueeze(1))   # (B, 1, C, H, W)
    return tuple(out)
