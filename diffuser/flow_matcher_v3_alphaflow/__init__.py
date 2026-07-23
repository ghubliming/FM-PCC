"""Compatibility namespace for α-Flow (Gen3v7) configs loaded through diffuser.*.

diffuser/utils/config.py:import_class() hard-prefixes every config class string with the
repo package name ('diffuser'), so 'flow_matcher_v3_alphaflow.models.AlphaFlowODE' is
looked up as 'diffuser.flow_matcher_v3_alphaflow.models.AlphaFlowODE'. This shim forwards
to the real top-level package. Mirrors diffuser/flow_matcher_v3_meanflow/ (Gen3v6, fix_1)
and diffuser/flow_matcher_v3_imeanflow/ (Gen3v4).

🔴 This is the FIFTH location of the copy-modify checklist, and the one Gen3v6 forgot
(Gen3v6 fix_1 §8): the dependency is invisible until diffuser.utils.Config is actually
called with a string class, which only happens on the TRAIN path.
"""
