"""Concrete evaluators, one package per DataEval family.

Each package's ``__init__`` may re-export its ``params`` module: light, and already
imported by ``config.schemas``. It never re-exports its evaluator module, which imports
DataEval and would pull the orchestration layer into config loading.
"""

__all__ = ["quality"]

from dataeval_flow.evaluators import quality
