import numpy as np
import biorbd
import casadi as cas
from time import time
import smtplib, ssl
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Bounds,
    ConstraintFcn,
    ObjectiveFcn,
    BiMapping,
    ConstraintList,
    InitialGuessList,
    InterpolationType,
    ObjectiveList,
    Node,
    DynamicsList,
    BoundsList,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path: str, final_time: float, n_shooting: int, n_threads: int) -> OptimalControlProgram:
    """
    Prepare the Euler version of the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The initial guess for the time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    np.random.seed(0)
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque() - biorbd_model.nbRoot()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=n_q + 5, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Initial guesses
    vz0 = 6.0
    x = np.vstack((np.zeros((n_q, n_shooting + 1)), np.ones((n_qdot, n_shooting + 1))))
    x[2, :] = (
        vz0 * np.linspace(0, final_time, n_shooting + 1) + -9.81 / 2 * np.linspace(0, final_time, n_shooting + 1) ** 2
    )
    x[3, :] = np.linspace(0, 2 * np.pi, n_shooting + 1)
    x[5, :] = np.linspace(0, 2 * np.pi, n_shooting + 1)
    x[6, :] = np.random.random((1, n_shooting + 1)) * np.pi - np.pi
    x[7, :] = np.random.random((1, n_shooting + 1)) * np.pi

    x[n_q + 2, :] = vz0 - 9.81 * np.linspace(0, final_time, n_shooting + 1)
    x[n_q + 3, :] = 2 * np.pi / final_time
    x[n_q + 5, :] = 2 * np.pi / final_time

    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.EACH_FRAME)

    # Path constraint
    x_bounds = BoundsList()
    x_min = np.zeros((n_q + n_qdot, 3))
    x_max = np.zeros((n_q + n_qdot, 3))
    x_min[:, 0] = [0, 0, 0, 0, 0, 0, -2.8, 2.8, -1, -1, 7, x[n_q + 3, 0], 0, x[n_q + 5, 0], 0, 0]
    x_max[:, 0] = [0, 0, 0, 0, 0, 0, -2.8, 2.8, 1, 1, 10, x[n_q + 3, 0], 0, x[n_q + 5, 0], 0, 0]
    x_min[:, 1] = [
        -1,
        -1,
        -0.001,
        -0.001,
        -np.pi / 4,
        -np.pi,
        -np.pi,
        0,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
    ]
    x_max[:, 1] = [1, 1, 5, 2 * np.pi + 0.001, np.pi / 4, 50, 0, np.pi, 100, 100, 100, 100, 100, 100, 100, 100]
    x_min[:, 2] = [
        -0.1,
        -0.1,
        -0.1,
        2 * np.pi - 0.1,
        -15 * np.pi / 180,
        2 * np.pi,
        -np.pi,
        0,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
    ]
    x_max[:, 2] = [
        0.1,
        0.1,
        0.1,
        2 * np.pi + 0.1,
        15 * np.pi / 180,
        20 * np.pi,
        0,
        np.pi,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    ]
    x_bounds.add(bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    u_mapping = BiMapping([None, None, None, None, None, None, 0, 1], [6, 7])

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.5, max_bound=1.5)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        n_threads=n_threads,
        tau_mapping=u_mapping,
        ode_solver=OdeSolver.RK4(),
    )


if __name__ == "__main__":
    root_folder = "/".join(__file__.split("/")[:-1])
    np.random.seed(0)
    n_threads = 8

    ocp = prepare_ocp("JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=100, n_threads=n_threads)

    tic = time()
    sol = ocp.solve(solver_options={"tol": 1e-15, "constr_viol_tol": 1e-15, "max_iter": 1000})
    print(f"Time to solve : {time() - tic}sec")

    port = 465  # For SSL
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("evidoux@gmail.com", "Josee9596")
        server.sendmail("evidoux@gmail.com", "evidoux@gmail.com", f'n_threads = {n_threads}\n Time to solve : {time() - tic}sec')




