import os

try:
    ncpu = os.cpu_count()
    if 'sched_getaffinity' in dir(os):
        ncpu = len(os.sched_getaffinity(0))
except:
    import multiprocessing
    ncpu = multiprocessing.cpu_count()

os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
import jax
if jax.device_count('cpu') != ncpu:
    raise ImportError(f"Importing JAX with 'XLA_FLAGS=--xla_force_host_platform_device_count={ncpu}' failed.")

from jax.config import config
config.update("jax_enable_x64", True)