"""All figure builders, with the figure group each writes into.

A builder is ``fn(outdir) -> (path, provenance) | None``. ``make_figs.py`` calls it
with ``../figures/<group>/``. Each environment module lists ``ALL = [(name, group,
builder), ...]``; the group must be one of ``make_figs.GROUPS``.
"""
from . import avoiding, exec_paths, expert, frontier, paths, scenes

ALL = (list(avoiding.ALL) + list(scenes.ALL) + list(paths.ALL) + list(expert.ALL)
       + list(frontier.ALL) + list(exec_paths.ALL))
