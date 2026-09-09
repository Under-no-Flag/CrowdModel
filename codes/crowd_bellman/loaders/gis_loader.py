"""Load versioned HP GIS square-cell scenes without QGIS or coordinate rescaling."""
from pathlib import Path
import hashlib
import json

import numpy as np

from ..compilers.config_compiler import CompiledSceneBundle, _corner_block_mask
from ..scenes import BaseScene, SimulationConfig


def load_gis_scene(version_dir, cfg: SimulationConfig | None = None):
    """Return (bundle, cfg, manifest). Targets are regions, never automatic sinks.

    Draft boundaries are available as named regions for review only. This loader
    does not turn them into external sources or authorize a physical experiment.
    """
    root = Path(version_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['schema_version'] != 1 or manifest['units'] != 'm'
            or manifest['crs'] != 'EPSG:32651'):
        raise ValueError('Unsupported GIS scene schema/units/CRS')
    payload = (root / 'grid.npz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['grid_sha256']:
        raise ValueError('GIS grid hash mismatch')
    ny, nx = manifest['shape_yx']
    dx = manifest['dx_m']
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError('Invalid physical grid spacing')
    if cfg is None:
        cfg = SimulationConfig(nx=nx, ny=ny, dx=dx)
    if (cfg.ny, cfg.nx) != (ny, nx) or cfg.dx != dx:
        raise ValueError('Configuration grid/scale differs from immutable GIS scene')
    with np.load(root / 'grid.npz', allow_pickle=False) as data:
        walk = data['walkable'].copy()
        if walk.dtype != bool or walk.shape != (ny, nx) or not walk.any():
            raise ValueError('Invalid walkable mask')
        regions = {item['key']: data[item['key']].copy()
                   for item in manifest['targets'] + manifest['boundaries']}
    if any(mask.shape != walk.shape or mask.dtype != bool or np.any(mask & ~walk)
           for mask in regions.values()):
        raise ValueError('Region masks must be subsets of the walkable domain')
    regions['walkable'] = walk.copy()
    zero = np.zeros(walk.shape)
    scene = BaseScene(walk, zero.copy(), np.zeros_like(walk), {}, {}, {}, ~walk,
                      zero.copy(), np.ones_like(zero), _corner_block_mask(walk),
                      0, 0, 0, (0, 0, 0), (0, 0))
    if cfg.wall_avoidance_weight != 0 or cfg.wall_clearance_cells != 0:
        raise ValueError('GIS grid has frozen walls; additional wall geometry/cost is unsupported')
    return CompiledSceneBundle(scene, regions, {}, {}, {}), cfg, manifest
