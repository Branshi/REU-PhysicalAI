from typing import Any

from common.rollout.types import AccelerationFn, DynamicsFn, StateTensor


def euler_step(
    state: StateTensor,
    dynamics_fn: DynamicsFn,
    dt: float,
) -> StateTensor:
    return state + dt * dynamics_fn(state)


def rk4_step(
    state: StateTensor,
    dynamics_fn: DynamicsFn,
    dt: float,
) -> StateTensor:
    k1 = dynamics_fn(state)
    k2 = dynamics_fn(state + 0.5 * dt * k1)
    k3 = dynamics_fn(state + 0.5 * dt * k2)
    k4 = dynamics_fn(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def velocity_verlet_step(
    # positions has shape like [num_bodies, dim] or [batch, num_bodies, dim].
    positions: StateTensor,
    velocities: StateTensor,
    # acceleration_fn computes acceleration from positions and any extra physics arguments.
    acceleration_fn: AccelerationFn,
    dt: float,
    # *acceleration_args collects extra positional arguments for acceleration_fn.
    # Example: masses in compute_acceleration(positions, masses, ...).
    *acceleration_args: Any,
    # acceleration is optional. If you already computed the current acceleration,
    # pass it in so the integrator does not compute it again.
    acceleration: StateTensor | None = None,
    # **acceleration_kwargs collects extra named arguments for acceleration_fn.
    # Example: G=1.0 and epsilon=0.15.
    **acceleration_kwargs: Any,
) -> tuple[StateTensor, StateTensor, StateTensor]:
    # If the caller did not provide the current acceleration, compute it at x_t.
    if acceleration is None:
        acceleration = acceleration_fn(
            positions,
            *acceleration_args,
            **acceleration_kwargs,
        )

    # Velocity Verlet position update:
    # x_{t+1} = x_t + v_t dt + 1/2 a_t dt^2
    next_positions = positions + velocities * dt + 0.5 * acceleration * dt**2

    # Compute acceleration at the new position x_{t+1}.
    next_acceleration = acceleration_fn(
        next_positions,
        *acceleration_args,
        **acceleration_kwargs,
    )

    # Velocity Verlet velocity update:
    # v_{t+1} = v_t + 1/2 (a_t + a_{t+1}) dt
    next_velocities = velocities + 0.5 * (acceleration + next_acceleration) * dt

    # Return the new state and the new acceleration.
    # Returning next_acceleration lets the caller reuse it on the next step.
    return next_positions, next_velocities, next_acceleration
