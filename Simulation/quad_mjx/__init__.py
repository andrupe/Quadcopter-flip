"""
Vectorised MJX/JAX quadcopter environment for the Quadcopter-flip project.

This package is a faithful PORT of ``Simulation/quad_flip_env.py`` (observation contract,
reward, termination, sensor model, domain randomisation, reference repertoire) onto
MuJoCo-MJX, not a re-specification of it.  See ``spec.PORT_NOTES`` for the deliberate
deviations -- there is exactly one, and it is the removal of the previous port's
accidental ones.

Import order note: modules import each other with relative imports, so this package must
be imported as ``quad_mjx`` with ``Simulation/`` on ``sys.path``.
"""

from . import spec, dynamics, lighthouse

__all__ = ["spec", "dynamics", "lighthouse"]
