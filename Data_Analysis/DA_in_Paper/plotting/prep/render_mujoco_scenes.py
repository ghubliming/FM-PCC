#!/usr/bin/env python3
"""Render the quadrotor scenes with MuJoCo itself, for Chapter 5.

    MUJOCO_GL=osmesa <python with mujoco> plotting/prep/render_mujoco_scenes.py [--check]

Why this replaces frames cut from rollout GIFs. Those frames were 140x140 px,
mostly black, and carried the diagnostic step counter burned into the image. A
figure that introduces an environment should show the environment, so it is
rendered here instead, from the same inputs the simulator uses:

* the scene     d3il/environments/d3il/models/mj/robot/quadrotor/scenes/<scene>.xml,
                which includes quadrotor_modified.xml and its Skydio X2 mesh;
* the vehicle   placed at the start of the reference path, level, at the path's
                altitude -- a pose, not a simulated state;
* the path      sampled from the demonstration generator itself,
                uav_expert_data_collect/trajectories.py, and drawn as a thin tube.

Nothing here is simulated and no model is run: mj_forward only places the bodies.

Needs the `mujoco` wheel and an offscreen GL (OSMesa works without a GPU). The
builders stay stdlib-only, so the output goes to data/prepared/ and is copied into
the store by make_figs.py through sources.VENDORED, like the other prepared files.
Every render parameter is declared in sources.MUJOCO_RENDERS and stamped in
data/prepared/MUJOCO_RENDERS.json, so a changed camera is detected as stale.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import sources as S                                            # noqa: E402

STAMP = os.path.join(S.PREPARED, 'MUJOCO_RENDERS.json')


def reference_path(spec, n=160):
    """Sample the demonstration generator's own reference path for this scene."""
    import numpy as np
    if spec['path_fn'] == 'avoiding_demo':
        # v3.68c: UAV-pillars flies the avoiding planner; its "demonstration" is a D3IL-avoiding
        # demonstration mapped into the arena by uav_avoiding_bridge/frame.py (scale 36).
        import json as _json
        sys.path.insert(0, S.REPO)
        from uav_avoiding_bridge import frame as F
        idx, alt = spec['path_args']
        sc = _json.load(open(S.AVOIDING_SCENE))
        xy = sc['demonstrations'][idx]
        return [np.array([*F.to_world_xy(x, y), alt]) for x, y in xy]
    sys.path.insert(0, os.path.join(S.REPO, 'uav_expert_data_collect'))
    import trajectories as T
    fn = getattr(T, spec['path_fn'])
    f = fn(*spec['path_args'])
    ts = np.linspace(0.0, spec['path_args'][-1], n)
    return [np.asarray(f(t)[0] if isinstance(f(t), tuple) else f(t))[:3] for t in ts]


def add_tube(scene, pts, rgba, radius=0.018):
    """Append the path to the scene as capsules between consecutive samples."""
    import mujoco
    import numpy as np
    for a, b in zip(pts, pts[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        if np.linalg.norm(b - a) < 1e-6:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        scene.ngeom += 1


def render(name, spec):
    import mujoco
    import numpy as np
    from PIL import Image

    xml = os.path.join(S.REPO, S.UAV_SCENE_DIR, spec['scene'])
    m = mujoco.MjModel.from_xml_path(xml)
    w, h = spec['size']
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    m.vis.quality.shadowsize = 8192
    d = mujoco.MjData(m)

    path = reference_path(spec)
    start = path[0]
    # freejoint qpos: position then quaternion (w, x, y, z). The body's own quat in
    # the XML orients the mesh; a level vehicle keeps it.
    d.qpos[:3] = start
    d.qpos[3:7] = m.body('x2').quat
    mujoco.mj_forward(m, d)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = spec['lookat']
    cam.distance, cam.azimuth, cam.elevation = spec['distance'], spec['azimuth'], spec['elevation']

    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera=cam)
    add_tube(r.scene, path, spec['path_rgba'], radius=spec.get('tube_radius', 0.018))
    img = Image.fromarray(r.render())
    dst = os.path.join(S.PREPARED, name + '.png')
    img.save(dst)
    return dst, img.size


def render_platform(name, spec):
    """A model on its own: no scene, background removed via the segmentation pass."""
    import mujoco
    import numpy as np
    from PIL import Image, ImageFilter

    model = spec['model'] if os.path.isabs(spec['model']) else os.path.join(S.REPO, spec['model'])
    if spec.get('wrap_include'):
        xml = (f'<mujoco><compiler angle="radian" meshdir="{spec["meshdir"]}"/>'
               f'<visual><headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/></visual>'
               f'<include file="{model}"/></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
    else:
        m = mujoco.MjModel.from_xml_path(model)
    w, h = spec['size']
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, w), max(m.vis.global_.offheight, h)
    d = mujoco.MjData(m)
    for joint, q in spec.get('qpos_by_joint', {}).items():
        d.qpos[m.jnt_qposadr[m.joint(joint).id]] = q
    if 'free_body_pos' in spec:
        d.qpos[:3] = spec['free_body_pos']
    mujoco.mj_forward(m, d)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = spec['lookat']
    cam.distance, cam.azimuth, cam.elevation = spec['distance'], spec['azimuth'], spec['elevation']
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera=cam)
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    fg = r.render()[..., 0] >= 0                     # -1 where the ray hits no geom
    out = np.full_like(rgb, 255)
    out[fg] = rgb[fg]
    mask = Image.fromarray((fg * 255).astype('uint8'))
    edge = np.asarray(mask.filter(ImageFilter.MaxFilter(5))) > 0
    out[edge & ~fg] = (150, 150, 150)                # thin contour outside the silhouette
    ys, xs = np.where(edge)
    pad = 24
    out = out[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
    dst = os.path.join(S.PREPARED, name + '.png')
    Image.fromarray(out).save(dst)
    return dst, (out.shape[1], out.shape[0])


def declared(spec):
    # Round-trip through JSON so nested tuples (e.g. the pillars route ('L','R','L')) compare equal
    # to what the stamp file stores; converting only top-level tuples left pillars always stale.
    return json.loads(json.dumps(spec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='report staleness, write nothing')
    ap.add_argument('--only', default=None, help='render only the names containing this substring')
    a = ap.parse_args()
    try:
        with open(STAMP) as fh:
            old = json.load(fh)
    except (OSError, ValueError):
        old = {}

    if a.check:
        allr = {**getattr(S, 'PLATFORM_RENDERS', {}), **S.MUJOCO_RENDERS}
        bad = [n for n, sp in allr.items()
               if old.get(n) != declared(sp) or not os.path.isfile(os.path.join(S.PREPARED, n + '.png'))]
        for n in bad:
            print(f'STALE  {n}')
        print(f'{len(allr) - len(bad)} of {len(allr)} MuJoCo renders are current')
        return 1 if bad else 0

    os.makedirs(S.PREPARED, exist_ok=True)
    for name, spec in getattr(S, 'PLATFORM_RENDERS', {}).items():
        if a.only and a.only not in name:
            continue
        dst, size = render_platform(name, spec)
        old[name] = declared(spec)
        print(f'wrote  data/prepared/{os.path.basename(dst)}  {size[0]}x{size[1]}  (platform)')
    for name, spec in S.MUJOCO_RENDERS.items():
        if a.only and a.only not in name:
            continue
        dst, size = render(name, spec)
        old[name] = declared(spec)
        print(f'wrote  data/prepared/{os.path.basename(dst)}  {size[0]}x{size[1]}  ({spec["scene"]})')
    with open(STAMP, 'w') as fh:
        json.dump(old, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
