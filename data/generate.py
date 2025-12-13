import jax
import jax.numpy         as jnp
import matplotlib.pyplot as plt
import seaborn           as sns
import xarray            as xr

import jax_cfd.base       as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral   as spectral

import dataclasses

if __name__ == "__main__":

  # Set simulation parameters
  viscosity    = 1e-3
  max_velocity = 7
  cfl          = 0.5
  final_time   = 25.
  init_kmax    = 4.
  x_min        = 0.
  x_max        = 2 * jnp.pi
  nx           = 256
  y_min        = x_min
  y_max        = x_max
  ny           = nx
  seed         = 42   # seed for random number generation
  smooth       = True # use anti-aliasing
  outer_steps  = 32   # nb of snapshots in generated dataset

  # Define grid
  grid = grids.Grid(
    (nx, ny),
    domain=((x_min, x_max), (y_min, y_max))
  )

  # Set time step
  dt = cfd.equations.stable_time_step(max_velocity, cfl, viscosity, grid)

  # Set time integration
  step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt
  )

  # Set number of time steps
  # between two consecutive snapshots
  # in the dataset that will be generated
  inner_steps = (final_time // dt) // outer_steps
  
  print(f"inner_steps: {inner_steps}")

  # Set trajectory function
  # to save snapshots
  trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps
  )

  # Set initial condition
  v0 = cfd.initial_conditions.filtered_velocity_field(
    jax.random.PRNGKey(seed), grid, max_velocity, init_kmax
  )
  vorticity0     = cfd.finite_differences.curl_2d(v0).data
  vorticity_hat0 = jnp.fft.rfftn(vorticity0)

  # Compute time series in spectral space
  _, trajectory = trajectory_fn(vorticity_hat0)

  # Convert time series to vorticity space
  vorticity_space = jnp.fft.irfftn(trajectory, axes=(1, 2))

  print("Completed simulation!")

  # Plot some snapshots
  # TODO: Prepend absolute path to 'data.png'
  coords   = jnp.arange(nx) * (x_max - x_min) / nx
  nsnaps   = min(outer_steps, 30)
  coordsxr = {
    'time': dt * jnp.arange(nsnaps) * inner_steps,
    'x': coords,
    'y': coords,
  }
  snaps = xr.DataArray(
    vorticity_space[0:nsnaps], 
    dims=["time", "x", "y"], coords=coordsxr
  ).plot.imshow(
    col='time', col_wrap=5, cmap=sns.cm.icefire, robust=True
  )
  snaps.fig.savefig("data.png", dpi=300, bbox_inches="tight")

  print("Completed plot!")

  # return vorticity_space, float(dt)
