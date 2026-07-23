"""Compatibility namespace for MeanFlow (Gen3v6) configs loaded through diffuser.*.

diffuser/utils/config.py:import_class() hard-prefixes every config class string with the
repo package name ('diffuser'), so 'flow_matcher_v3_meanflow.models.MeanFlowODE' is looked
up as 'diffuser.flow_matcher_v3_meanflow.models.MeanFlowODE'. This shim forwards to the
real top-level package. Mirrors diffuser/flow_matcher_v3_imeanflow/ (Gen3v4).
"""
