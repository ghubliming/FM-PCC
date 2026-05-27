import os
import yaml
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as _mpa
import matplotlib.patches as _mpa_hs
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3C

def _hs_xy_draw(ax, hs, enlarge, xlim, ylim):
    """Draw halfspace: shaded infeasible region + boundary line + feasible-side arrow."""
    pt1, pt2, side = hs
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    dx, dy = x2 - x1, y2 - y1
    nx, ny = (-dy, dx) if side == 'above' else (dy, -dx)   # normal → feasible side
    nlen = np.hypot(nx, ny)
    if nlen < 1e-9:
        return
    nx /= nlen;  ny /= nlen
    x1 += enlarge * nx;  y1 += enlarge * ny    # tightening shifts boundary inward
    x2 += enlarge * nx;  y2 += enlarge * ny
    dx, dy = x2 - x1, y2 - y1
    tx = sorted([(xlim[0]-x1)/dx, (xlim[1]-x1)/dx]) if abs(dx) > 1e-9 else [-1e9, 1e9]
    ty = sorted([(ylim[0]-y1)/dy, (ylim[1]-y1)/dy]) if abs(dy) > 1e-9 else [-1e9, 1e9]
    t_lo = max(tx[0], ty[0])
    t_hi = min(tx[1], ty[1])
    if t_lo >= t_hi:
        return
    px = [x1 + t_lo*dx, x1 + t_hi*dx]
    py = [y1 + t_lo*dy, y1 + t_hi*dy]

    _C = [(xlim[0],ylim[0]),(xlim[1],ylim[0]),(xlim[1],ylim[1]),(xlim[0],ylim[1])]
    def _inf(c): return (c[0]-x1)*nx + (c[1]-y1)*ny < -1e-9
    def _edge(p):
        if abs(p[1]-ylim[0]) < 1e-9: return 0
        if abs(p[0]-xlim[1]) < 1e-9: return 1
        if abs(p[1]-ylim[1]) < 1e-9: return 2
        return 3
    _e0, _e1 = _edge((px[0], py[0])), _edge((px[1], py[1]))
    _df = (_e1 - _e0) % 4
    if _df:
        _cf = [_C[(_e0+1+k)%4] for k in range(_df)]
        _cr = [_C[(_e0-k)%4]   for k in range(4-_df)]
        if _cf and all(_inf(c) for c in _cf):
            _poly = [(px[0],py[0])] + _cf + [(px[1],py[1])]
        elif _cr and all(_inf(c) for c in _cr):
            _poly = [(px[1],py[1])] + _cr + [(px[0],py[0])]
        else:
            _ifc = [c for c in _cf if _inf(c)] or [c for c in _cr if _inf(c)]
            _poly = ([(px[0],py[0])] + _ifc + [(px[1],py[1])]) if _ifc else None
        if _poly and len(_poly) >= 3:
            ax.add_patch(_mpa_hs.Polygon(_poly, closed=True, facecolor='darkorange',
                                          alpha=0.15, edgecolor='none', zorder=1))

    ax.plot(px, py, color='darkorange', linewidth=1.5, zorder=3, label='halfspace')
    mx, my = (px[0]+px[1])/2, (py[0]+py[1])/2
    ax.annotate('', xy=(mx+nx*0.06, my+ny*0.06), xytext=(mx, my),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
    ax.text(mx+nx*0.09, my+ny*0.09, 'feasible', fontsize=6, color='darkorange',
            ha='center', va='center')


def plot_geo_constraints(geo_name, geo_config, out_dir, is_tightened=False):
    out_path = os.path.join(out_dir, 'constraint_overview.png')
    if os.path.exists(out_path):
        return   # idempotent

    constraint_types = geo_config.get('constraint_types', [])
    enlarge = (geo_config.get('enlarge_constraints') or 0.0) if is_tightened else 0.0

    has_bounds = 'bounds' in constraint_types
    ws_lb = ws_ub = lb_d = ub_d = None
    _Z_DISP = (0.0, 0.50)   
    if has_bounds and 'workspace_bounds' in geo_config:
        ws_lb = np.array(geo_config['workspace_bounds']['lb'], dtype=float)
        ws_ub = np.array(geo_config['workspace_bounds']['ub'], dtype=float)
        ws_lb += enlarge;  ws_ub -= enlarge
        lb_d = ws_lb.copy();  ub_d = ws_ub.copy()
        lb_d[2] = _Z_DISP[0] if np.isinf(lb_d[2]) else lb_d[2]
        ub_d[2] = _Z_DISP[1] if np.isinf(ub_d[2]) else ub_d[2]

    def _xlim(): return (lb_d[0]-0.05, ub_d[0]+0.05) if lb_d is not None else (0.20, 0.80)
    def _ylim(): return (lb_d[1]-0.05, ub_d[1]+0.05) if lb_d is not None else (-0.45, 0.45)
    def _zlim(): return (lb_d[2]-0.02, ub_d[2]+0.05) if lb_d is not None else (_Z_DISP[0]-0.02, _Z_DISP[1]+0.05)

    halfspace_list = geo_config.get('halfspace_constraints', []) if 'halfspace' in constraint_types else []
    obstacle_list  = geo_config.get('obstacle_constraints',  []) if 'obstacles'  in constraint_types else []

    fig = plt.figure(figsize=(16, 5))
    _tstr = ' [tightened]' if is_tightened else ''
    fig.suptitle(f'{geo_name}{_tstr}  |  types: {constraint_types}',
                 fontsize=11, fontweight='bold', y=0.98)

    ax3 = fig.add_subplot(131, projection='3d')
    ax3.set_title('3D view', fontsize=9)
    ax3.set_xlabel('x (m)', fontsize=7, labelpad=2)
    ax3.set_ylabel('y (m)', fontsize=7, labelpad=2)
    ax3.set_zlabel('z (m)', fontsize=7, labelpad=2)
    ax3.tick_params(labelsize=6)

    if lb_d is not None:
        x0, y0, z0 = lb_d;  x1v, y1v, z1v = ub_d
        for xs, ys, zs in [
            ([x0,x1v],[y0,y0],[z0,z0]), ([x0,x1v],[y1v,y1v],[z0,z0]),
            ([x0,x1v],[y0,y0],[z1v,z1v]), ([x0,x1v],[y1v,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y1v],[z0,z0]), ([x1v,x1v],[y0,y1v],[z0,z0]),
            ([x0,x0],[y0,y1v],[z1v,z1v]), ([x1v,x1v],[y0,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y0],[z0,z1v]), ([x1v,x1v],[y0,y0],[z0,z1v]),
            ([x0,x0],[y1v,y1v],[z0,z1v]), ([x1v,x1v],[y1v,y1v],[z0,z1v]),
        ]:
            ax3.plot(xs, ys, zs, color='steelblue', alpha=0.7, lw=1.2)
        ax3.add_collection3d(_P3C([
            [(x0,y0,z0),(x1v,y0,z0),(x1v,y1v,z0),(x0,y1v,z0)],
            [(x0,y0,z1v),(x1v,y0,z1v),(x1v,y1v,z1v),(x0,y1v,z1v)],
            [(x0,y0,z0),(x0,y0,z1v),(x0,y1v,z1v),(x0,y1v,z0)],
            [(x1v,y0,z0),(x1v,y0,z1v),(x1v,y1v,z1v),(x1v,y1v,z0)],
            [(x0,y0,z0),(x1v,y0,z0),(x1v,y0,z1v),(x0,y0,z1v)],
            [(x0,y1v,z0),(x1v,y1v,z0),(x1v,y1v,z1v),(x0,y1v,z1v)],
        ], alpha=0.06, facecolor='steelblue', edgecolor='none'))

    for obs in obstacle_list:
        cx, cy = float(obs['center'][0]), float(obs['center'][1])
        obs_dims = obs.get('dimensions', ['x', 'y'])
        cz = (float(obs['center'][2]) if len(obs['center']) > 2 and 'z' in obs_dims
              else ((lb_d[2]+ub_d[2])/2 if lb_d is not None else 0.12))
        r = obs['radius'] + enlarge
        u = np.linspace(0, 2*np.pi, 25);  v = np.linspace(0, np.pi, 15)
        ax3.plot_surface(
            cx + r*np.outer(np.cos(u), np.sin(v)),
            cy + r*np.outer(np.sin(u), np.sin(v)),
            cz + r*np.outer(np.ones_like(u), np.cos(v)),
            color='tomato', alpha=0.25, linewidth=0)

    _hs_zlo = lb_d[2] if lb_d is not None else _Z_DISP[0]
    _hs_zhi = ub_d[2] if ub_d is not None else _Z_DISP[1]
    for _hs3 in halfspace_list:
        _hpt1, _hpt2, _hside = _hs3
        _hx1, _hy1 = float(_hpt1[0]), float(_hpt1[1])
        _hx2, _hy2 = float(_hpt2[0]), float(_hpt2[1])
        _hdx, _hdy = _hx2-_hx1, _hy2-_hy1
        _hn = np.array([-_hdy, _hdx]) if _hside == 'above' else np.array([_hdy, -_hdx])
        _hnl = float(np.hypot(*_hn))
        if _hnl < 1e-9: continue
        _hn /= _hnl
        _hx1 += enlarge*_hn[0]; _hy1 += enlarge*_hn[1]
        _hx2 += enlarge*_hn[0]; _hy2 += enlarge*_hn[1]
        _hdx, _hdy = _hx2-_hx1, _hy2-_hy1
        _hxl, _hyl = _xlim(), _ylim()
        _htx = sorted([(_hxl[0]-_hx1)/_hdx, (_hxl[1]-_hx1)/_hdx]) if abs(_hdx) > 1e-9 else [-1e9, 1e9]
        _hty = sorted([(_hyl[0]-_hy1)/_hdy, (_hyl[1]-_hy1)/_hdy]) if abs(_hdy) > 1e-9 else [-1e9, 1e9]
        _htlo = max(_htx[0], _hty[0]); _hthi = min(_htx[1], _hty[1])
        if _htlo >= _hthi: continue
        _hpx = [_hx1+_htlo*_hdx, _hx1+_hthi*_hdx]
        _hpy = [_hy1+_htlo*_hdy, _hy1+_hthi*_hdy]
        ax3.add_collection3d(_P3C([[
            [_hpx[0], _hpy[0], _hs_zlo], [_hpx[1], _hpy[1], _hs_zlo],
            [_hpx[1], _hpy[1], _hs_zhi], [_hpx[0], _hpy[0], _hs_zhi],
        ]], alpha=0.25, facecolor='darkorange', edgecolor='darkorange', lw=0.8))

    if not has_bounds and not obstacle_list and not halfspace_list:
        ax3.text2D(0.5, 0.5, 'no geometric\nconstraints',
                   ha='center', va='center', transform=ax3.transAxes, fontsize=9, color='gray')

    ax3.set_xlim(*_xlim());  ax3.set_ylim(*_ylim());  ax3.set_zlim(*_zlim())

    ax_xy = fig.add_subplot(132)
    ax_xy.set_title('XY top-down (z projected)', fontsize=9)
    ax_xy.set_xlabel('x forward (m)', fontsize=7)
    ax_xy.set_ylabel('y lateral (m)', fontsize=7)
    ax_xy.set_aspect('equal');  ax_xy.grid(True, linestyle='--', alpha=0.4)
    ax_xy.tick_params(labelsize=6)

    if lb_d is not None:
        ax_xy.add_patch(_mpa.Rectangle(
            (lb_d[0], lb_d[1]), ub_d[0]-lb_d[0], ub_d[1]-lb_d[1],
            lw=1.5, edgecolor='steelblue', facecolor='steelblue', alpha=0.12, label='bounds'))
    else:
        ax_xy.text(0.5, 0.5, 'no bounds', ha='center', va='center',
                   transform=ax_xy.transAxes, fontsize=9, color='gray')

    for hs in halfspace_list:
        _hs_xy_draw(ax_xy, hs, enlarge, _xlim(), _ylim())
    for obs in obstacle_list:
        ax_xy.add_patch(_mpa.Circle(
            (float(obs['center'][0]), float(obs['center'][1])), obs['radius']+enlarge,
            lw=1.5, edgecolor='tomato', facecolor='tomato', alpha=0.2, label='obstacle'))
        ax_xy.plot(float(obs['center'][0]), float(obs['center'][1]), 'r+', ms=6)

    ax_xy.set_xlim(*_xlim());  ax_xy.set_ylim(*_ylim())

    ax_xz = fig.add_subplot(133)
    ax_xz.set_title('XZ side (y projected)', fontsize=9)
    ax_xz.set_xlabel('x forward (m)', fontsize=7)
    ax_xz.set_ylabel('z vertical (m)', fontsize=7)
    ax_xz.grid(True, linestyle='--', alpha=0.4)
    ax_xz.tick_params(labelsize=6)

    if lb_d is not None:
        ax_xz.add_patch(_mpa.Rectangle(
            (lb_d[0], lb_d[2]), ub_d[0]-lb_d[0], ub_d[2]-lb_d[2],
            lw=1.5, edgecolor='steelblue', facecolor='steelblue', alpha=0.12))
        z_inf_lo = np.isinf(ws_lb[2])
        z_inf_hi = np.isinf(ws_ub[2])
        ax_xz.axhline(lb_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7,
                      label='z=−∞ (display clamped)' if z_inf_lo else f'floor z={lb_d[2]:.3f} m')
        ax_xz.axhline(ub_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7,
                      label='z=+∞ (display clamped)' if z_inf_hi else f'ceiling z={ub_d[2]:.3f} m')
        ax_xz.legend(fontsize=6, loc='upper right')
    else:
        ax_xz.text(0.5, 0.5, 'no bounds', ha='center', va='center',
                   transform=ax_xz.transAxes, fontsize=9, color='gray')

    for obs in obstacle_list:
        obs_dims = obs.get('dimensions', ['x', 'y'])
        cx_o = float(obs['center'][0])
        r_o  = obs['radius'] + enlarge
        if 'z' in obs_dims and len(obs['center']) > 2:
            ax_xz.add_patch(_mpa.Circle(
                (cx_o, float(obs['center'][2])), r_o,
                lw=1.5, edgecolor='tomato', facecolor='tomato', alpha=0.2))
        else:
            # 2D obstacle (xy-only): show as circle at workspace z-midpoint
            _cz_mid = (lb_d[2] + ub_d[2]) / 2 if lb_d is not None else 0.20
            ax_xz.add_patch(_mpa.Circle(
                (cx_o, _cz_mid), r_o,
                lw=1.2, edgecolor='tomato', facecolor='tomato', alpha=0.25,
                linestyle='--'))

    for _hs_xz in halfspace_list:
        _hxpt1, _hxpt2, _hxside = _hs_xz
        _hxx1, _hxy1 = float(_hxpt1[0]), float(_hxpt1[1])
        _hxx2, _hxy2 = float(_hxpt2[0]), float(_hxpt2[1])
        _hxdx, _hxdy = _hxx2 - _hxx1, _hxy2 - _hxy1
        _hxnx, _hxny = (-_hxdy, _hxdx) if _hxside == 'above' else (_hxdy, -_hxdx)
        _hxnl = float(np.hypot(_hxnx, _hxny))
        if _hxnl < 1e-9:
            continue
        _hxnx /= _hxnl;  _hxny /= _hxnl
        _hxx1 += enlarge * _hxnx;  _hxy1 += enlarge * _hxny
        _hxx2 += enlarge * _hxnx;  _hxy2 += enlarge * _hxny
        _hxdx, _hxdy = _hxx2 - _hxx1, _hxy2 - _hxy1
        _hxxl, _hxyl = _xlim(), _ylim()
        _hxtx = sorted([(_hxxl[0]-_hxx1)/_hxdx, (_hxxl[1]-_hxx1)/_hxdx]) if abs(_hxdx) > 1e-9 else [-1e9, 1e9]
        _hxty = sorted([(_hxyl[0]-_hxy1)/_hxdy, (_hxyl[1]-_hxy1)/_hxdy]) if abs(_hxdy) > 1e-9 else [-1e9, 1e9]
        _hxtlo = max(_hxtx[0], _hxty[0]);  _hxthi = min(_hxtx[1], _hxty[1])
        if _hxtlo >= _hxthi:
            continue
        _hxpx = [_hxx1 + _hxtlo*_hxdx, _hxx1 + _hxthi*_hxdx]
        _hxpy = [_hxy1 + _hxtlo*_hxdy, _hxy1 + _hxthi*_hxdy]
        _xb_lo = min(_hxpx[0], _hxpx[1])
        _xb_hi = max(_hxpx[0], _hxpx[1])
        _yb_lo = min(_hxpy[0], _hxpy[1])
        _yb_hi = max(_hxpy[0], _hxpy[1])
        ax_xz.axvspan(_xb_lo, _xb_hi, color='darkorange', alpha=0.13, zorder=1)
        ax_xz.axvline(_xb_lo, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
        ax_xz.axvline(_xb_hi, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
        _zc = (_zlim()[0] + _zlim()[1]) / 2
        ax_xz.text((_xb_lo + _xb_hi) / 2, _zc,
                   f'HS\ny∈[{_yb_lo:.2f},{_yb_hi:.2f}]',
                   ha='center', va='center', fontsize=5, color='darkorange',
                   style='italic', zorder=3)

    ax_xz.set_xlim(*_xlim());  ax_xz.set_ylim(*_zlim())

    if 'dynamics' in constraint_types:
        fig.text(0.5, 0.01,
                 'Dynamics: c_pos[t+1] = c_pos[t] + act[t]  (Euler link — no geometric shape)',
                 ha='center', fontsize=7, color='dimgray', style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[ geo ] Constraint overview → {out_path}')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../config/visual_aligning_eval.yaml')
    out_base_dir = os.path.join(script_dir, 'constraint_plots')
    
    print(f"Loading config from: {os.path.abspath(config_path)}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    geo_variants = config.get('geo_constraint_variants', [])
    if not geo_variants:
        print("No geo_constraint_variants found in config.")
        return
        
    print(f"Found {len(geo_variants)} geometric constraint variants.")
    enlarge = config.get('enlarge_constraints', 0.0)
    
    for geo_config in geo_variants:
        geo_name = geo_config.get('name', 'unnamed')
        out_dir = os.path.join(out_base_dir, geo_name)
        os.makedirs(out_dir, exist_ok=True)
        
        if enlarge:
            geo_config['enlarge_constraints'] = enlarge
            
        print(f"Plotting constraints for: {geo_name}")
        out_file = os.path.join(out_dir, 'constraint_overview.png')
        if os.path.exists(out_file):
            os.remove(out_file)
            
        plot_geo_constraints(geo_name, geo_config, out_dir, is_tightened=False)
        
        if enlarge and enlarge > 0.0:
            c_types = geo_config.get('constraint_types', [])
            if 'bounds' in c_types or 'obstacles' in c_types or 'halfspace' in c_types:
                geo_name_tightened = f"{geo_name}-tightened"
                out_dir_tightened = os.path.join(out_base_dir, geo_name_tightened)
                os.makedirs(out_dir_tightened, exist_ok=True)
                
                print(f"Plotting constraints for: {geo_name_tightened}")
                out_file_tight = os.path.join(out_dir_tightened, 'constraint_overview.png')
                if os.path.exists(out_file_tight):
                    os.remove(out_file_tight)
                    
                plot_geo_constraints(geo_name_tightened, geo_config, out_dir_tightened, is_tightened=True)

    print(f"\nAll plots saved to: {out_base_dir}")

if __name__ == '__main__':
    main()
