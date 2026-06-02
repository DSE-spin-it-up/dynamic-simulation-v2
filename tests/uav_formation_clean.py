"""
Simple prototype: N fixed-wing UAVs (3-DOF point-mass model) cooperatively
carrying a cable-suspended payload along a reference trajectory while holding
formation.

Same physics as the fancier version (fixed-wing wind-axis dynamics + rigid
cables + payload), kept in the SIMPLE style of the original prototype: explicit
per-timestep `subject_to` loops, one variable list per UAV, and an additive cost
breakdown. No `.map()`.

Two solve strategies share the SAME per-window NLP builder (`build_nlp`):

  * 'monolithic' — one big NLP over all N nodes (the original prototype).
  * 'rhc'        — stitched receding horizon. Solve a short window of N_h nodes,
    commit the first N_apply, slide forward, warm-start from the shifted previous
    solution. Each NLP is small and the same size every window, so cost is roughly
    linear in N and stays well-conditioned. A terminal tracking weight (Q_term)
    keeps the short horizon from cutting corners.

Each UAV is modelled in wind / flight-path axes (flat, non-rotating earth):

    State   x = (V, gamma, chi, p_n, p_e, h)
    Control u = (T, alpha, mu)

      m V_dot            = T cos a - D - mg sin(g)          + F_ext . t_hat
      m V gamma_dot      = (L + T sin a) cos(mu) - mg cos g + F_ext . n_hat
      m V cos(g) chi_dot = (L + T sin a) sin(mu)            + F_ext . h_hat
      p_n_dot = V cos(chi) cos(gamma)
      p_e_dot = V sin(chi) cos(gamma)
      h_dot   = V sin(gamma)

Payload: a point mass hanging from N_uav inextensible cables of length L_c, one
attached to each UAV. Cables are assumed always taut (rigid links):

      ||p_i - p_L|| = L_c           (length constraint, every node)
      Tc_i >= 0                     (cables pull, never push)
      F_ext on UAV i  = -Tc_i u_hat_i ;  u_hat_i = (p_i - p_L)/||p_i - p_L||
      m_L a_L = sum_i Tc_i u_hat_i - m_L g e_z      (Newton for the load)

Objective: track the reference with the PAYLOAD, keep the UAVs spread in
formation, penalise control effort and thrust energy.
"""

import time

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Solve strategy ────────────────────────────────────────────────────────────

SOLVE_MODE = 'rhc'        # 'rhc' (stitched receding horizon) or 'monolithic'
N_h        = 50           # RHC prediction-horizon length [nodes]
N_apply    = 10           # RHC nodes committed per solve before sliding the window
Q_term     = 100.0        # terminal payload-tracking weight (short horizons need this)

# ── Problem parameters ────────────────────────────────────────────────────────

N_uav   = 3          # number of UAVs (change freely)
N       = 600        # number of timesteps  (total time = (N-1)*dt)
dt      = 0.05       # timestep size [s]
M_EULER = 2          # Euler sub-steps per interval (accuracy vs. cost)

# Vehicle / aero parameters
m    = 6.8           # mass of each UAV [kg]
g    = 9.81          # gravity [m/s^2]
rho  = 1.225         # air density [kg/m^3]
S    = 1.4           # wing reference area [m^2]
CL0  = 0.20
AR=6.5
e=0.85
CLa  = 4.6           # lift-curve slope [1/rad]
CD0_drone  = 0.02
CD0_payload=1.07
S_payload=0.56


# State / control limits
V_min, V_max         = 10.0, 30.0
gam_max              = np.deg2rad(45.0)
T_min, T_max         = 0.0, 500.0
alpha_min, alpha_max = np.deg2rad(-15.0), np.deg2rad(15.0)
mu_max               = np.deg2rad(35.0)
d_min                = 2.0           # min distance between any two UAVs [m]
V_cruise             = 20.0

# Payload / cable parameters
m_L       = 60            # payload mass [kg]
cable_len = 12.5             # cable length L_c [m] (nominal)
cable_tol = 0.1              # allowed half-band on the chord length [m]
Tc_max    = 1000.0            # max cable tension [N]
e_z       = np.array([0.0, 0.0, 1.0])

# Objective weights
Q_track, Q_form = 50.0, 0
R_T, R_a, R_mu  = 0.0, 0, 0
W_energy        = 0   # weight on thrust energy sum(T * V * dt)
track_eps       = 0     # smooths linear tracking cost for IPOPT
R_dV            = 0   # penalizes UAV airspeed fluctuations between nodes
R_eq_T          = 0    # penalizes thrust mismatch between UAVs
R_eq_Tc         = 0    # penalizes cable-tension mismatch between UAVs
# Cable-tension regularization. The force split among N_uav cables on a single
# point payload is under-determined, so without these the solver picks bang-bang
# (0<->Tc_max) tensions that snap node to node. R_Tc penalizes magnitude (selects
# the smooth minimum-norm split); R_dTc penalizes the tension RATE (anti-snap) and
# also ties each RHC window's first tension to the previous window's value.
R_Tc, R_dTc = 1e-4, 5e-1
# Soft cable-length band: penalty on the slack that relaxes the chord-length
# inequality (see path constraints). Large => band nearly hard; small => loose.
W_cab = 1e3

# Formation: UAVs orbit on a cone above the payload (half-angle cone_ang from
# vertical). r_form is the horizontal orbit radius; depth is the vertical drop to
# the payload, so sqrt(r_form^2 + depth^2) == cable_len exactly (taut cable).
cone_ang = np.deg2rad(75.0)             # cone half-angle from the vertical
r_form   = cable_len * np.sin(cone_ang)
depth    = cable_len * np.cos(cone_ang)             # payload drop below UAVs
angles  = np.linspace(0, 2 * np.pi, N_uav, endpoint=False)
offsets = np.array([[r_form * np.cos(a), r_form * np.sin(a), 0.0]
                    for a in angles])               # (N_uav, 3)
assert cable_len > r_form, "cable must be longer than the formation radius"

# Orbit: the whole formation spins about the vertical axis through the payload at
# om_orbit while the payload climbs. V_orbit is the UAV horizontal orbital speed;
# its turn radius is r_form, so the required bank atan(V_orbit^2/(g*r_form)) must
# stay under mu_max — that's what caps how tight (small r_form) the orbit can be.
V_orbit  = 18                         # UAV horizontal orbital speed [m/s]
om_orbit = V_orbit / r_form             # formation spin rate [rad/s]
vY_final_tol = 0.2                      # terminal payload y-speed tolerance [m/s]
vY_final_bigM = 1e3                     # disables terminal y-speed constraint when mask=0

# ── Reference trajectory for the PAYLOAD (vertical climb, rounded turn, horizontal move) ─

t_vec  = np.linspace(0, (N - 1) * dt, N)
h0, climb = 100.0, 3.33                             # start height, climb rate [m/s]
y_speed = 25                                    # forward speed [m/s]
turn_radius = 8.0                               # circular fillet radius at climb/forward corner [m]

# Split trajectory: first quarter go up (z), then blend into forward (y) with
# a quarter-circle fillet instead of a sharp corner.
N_mid = N // 4
r_ref = np.zeros((N, 3))

# Original hard-corner point. The fillet is tangent to the vertical climb at
# z_peak - turn_radius and tangent to the horizontal leg at y = turn_radius.
z_peak = h0 + climb * t_vec[N_mid - 1]
turn_radius = min(turn_radius, 0.95 * (z_peak - h0))
arc_len = 0.5 * np.pi * turn_radius
s_turn_start = z_peak - h0 - turn_radius
s_turn_end = s_turn_start + arc_len

# Integrate along-path distance with a smooth speed ramp through the turn.
s_ref = np.zeros(N)
for k in range(1, N):
    q = np.clip((s_ref[k - 1] - s_turn_start) / arc_len, 0.0, 1.0)
    q = q * q * (3.0 - 2.0 * q)
    path_speed = climb + (y_speed - climb) * q
    s_ref[k] = s_ref[k - 1] + path_speed * dt

for k, s in enumerate(s_ref):
    if s <= s_turn_start:
        r_ref[k, 2] = h0 + s
    elif s <= s_turn_end:
        theta = np.pi - 0.5 * np.pi * ((s - s_turn_start) / arc_len)
        r_ref[k, 1] = turn_radius + turn_radius * np.cos(theta)
        r_ref[k, 2] = z_peak - turn_radius + turn_radius * np.sin(theta)
    else:
        r_ref[k, 1] = turn_radius + (s - s_turn_end)
        r_ref[k, 2] = z_peak

vel_ref = np.gradient(r_ref, t_vec, axis=0)

# Spin the formation about the vertical axis at om_orbit. offsets_rot[0] == offsets
# so the IC stays consistent; the UAVs trace circles of radius r_form as they climb.
def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
offsets_rot = np.stack([offsets @ _Rz(om_orbit * t_vec[k]).T
                        for k in range(N)])       # (N, N_uav, 3)
# Formation tracking schedule: keep rotating formation during climb,
# disable formation-shape tracking during forward line phase.
form_w = np.ones(N)
form_w[N_mid:] = 0.0


def uav_ref(k, i):
    """Reference state (V, gamma, chi, position) of UAV i at node k while orbiting
    the vertical axis at om_orbit and moving with the payload reference."""
    a = angles[i] + om_orbit * t_vec[k]
    p = r_ref[k] + offsets_rot[k, i] + depth * e_z
    v = om_orbit * r_form * np.array([-np.sin(a), np.cos(a), 0.0]) + vel_ref[k]
    V = np.linalg.norm(v)
    return V, np.arcsin(v[2] / V), np.arctan2(v[1], v[0]), p


# ── Dynamics helpers ──────────────────────────────────────────────────────────

def drone_rhs(x, u, F_ext):
    """3-DOF fixed-wing point-mass dynamics with external force F_ext (CasADi)."""
    V, gamma, chi = x[0], x[1], x[2]
    T, alpha, mu  = u[0], u[1], u[2]

    q = 0.5 * rho * V**2 * S
    L = q * (CL0 + CLa * alpha)
    D = q * (CD0 + CDa * alpha + CDa2 * alpha**2)

    t_hat = ca.vertcat(ca.cos(gamma) * ca.cos(chi),
                       ca.cos(gamma) * ca.sin(chi), ca.sin(gamma))
    n_hat = ca.vertcat(-ca.sin(gamma) * ca.cos(chi),
                       -ca.sin(gamma) * ca.sin(chi), ca.cos(gamma))
    h_hat = ca.vertcat(-ca.sin(chi), ca.cos(chi), 0.0)
    Ft, Fn, Fh = ca.dot(F_ext, t_hat), ca.dot(F_ext, n_hat), ca.dot(F_ext, h_hat)

    V_dot     = (T * ca.cos(alpha) - D) / m - g * ca.sin(gamma) + Ft / m
    gamma_dot = ((L + T * ca.sin(alpha)) * ca.cos(mu) - m * g * ca.cos(gamma)
                 + Fn) / (m * V)
    chi_dot   = ((L + T * ca.sin(alpha)) * ca.sin(mu) + Fh) \
                / (m * V * ca.cos(gamma))
    pn_dot = V * ca.cos(chi) * ca.cos(gamma)
    pe_dot = V * ca.sin(chi) * ca.cos(gamma)
    h_dot  = V * ca.sin(gamma)
    return ca.vertcat(V_dot, gamma_dot, chi_dot, pn_dot, pe_dot, h_dot)


def drone_vel(x):
    """Inertial velocity vector of a UAV: v = V * t_hat (CasADi)."""
    V, gamma, chi = x[0], x[1], x[2]
    return ca.vertcat(V * ca.cos(gamma) * ca.cos(chi),
                      V * ca.cos(gamma) * ca.sin(chi),
                      V * ca.sin(gamma))


def coupled_step(xs, us, Tcs, pL, vL):
    """One forward-Euler step (M_EULER sub-steps) of the WHOLE coupled system:
    all UAVs + the payload, with cable tensions Tcs held constant over dt.
    Each sub-step recomputes the cable directions from the current positions."""
    h = dt / M_EULER
    for _ in range(M_EULER):
        F_pay = ca.vertcat(0.0, 0.0, -m_L * g)        # gravity on payload
        xs_next = []
        for i in range(N_uav):
            d     = xs[i][3:6] - pL
            u_hat = d / ca.sqrt(ca.dot(d, d) + 1e-9)   # payload -> UAV
            xs_next.append(xs[i] + h * drone_rhs(xs[i], us[i], -Tcs[i] * u_hat))
            F_pay = F_pay + Tcs[i] * u_hat             # reaction on payload
        pL_next = pL + h * vL
        vL_next = vL + h * (F_pay / m_L)
        xs, pL, vL = xs_next, pL_next, vL_next
    return xs, pL, vL


# ── Kinematically consistent initial state ───────────────────────────────────

def consistent_ic():
    """Full system state with the drones orbiting the vertical axis above the
    payload (each flying its circle tangentially) while the payload climbs.

    The UAVs' horizontal orbital velocities cancel by symmetry, so the payload
    rises purely vertically at the climb rate. For every UAV the relative
    velocity (v_i - v_L) is tangent to its circle and hence perpendicular to the
    cable, so ||p_i - p_L|| is instantaneously constant: the rigid cables are
    kinematically consistent. Returns (xs, pL, vL).
    """
    xs = []
    for i in range(N_uav):
        V, gam, chi, p = uav_ref(0, i)
        xs.append(np.array([V, gam, chi, *p]))
    return xs, r_ref[0].astype(float), np.array([0.0, 0.0, climb])


# ── Parametric NLP builder (shared by both solve modes) ───────────────────────

def build_nlp(N_win, print_level=0):
    """Build a window NLP of length N_win with the initial state and payload
    reference exposed as `opti.parameter`s, so it can be (re)solved cheaply by
    just setting values + warm-starting. Returns a dict of handles."""
    opti = ca.Opti()

    # Decision variables (per-UAV lists, in the simple original style)
    x    = [opti.variable(6, N_win) for _ in range(N_uav)]
    u    = [opti.variable(3, N_win) for _ in range(N_uav)]
    Tc   = [opti.variable(1, N_win) for _ in range(N_uav)]
    eps  = [opti.variable(1, N_win) for _ in range(N_uav)]  # cable-band slack >= 0
    posL = opti.variable(3, N_win)
    velL = opti.variable(3, N_win)
    pos  = [x[i][3:6, :] for i in range(N_uav)]

    # Parameters carried per window
    X0  = [opti.parameter(6) for _ in range(N_uav)]   # fixed UAV states at node 0
    pL0 = opti.parameter(3)                            # fixed payload position
    vL0 = opti.parameter(3)                            # fixed payload velocity
    Ref = opti.parameter(3, N_win)                     # payload position reference
    Off = [opti.parameter(3, N_win) for _ in range(N_uav)]  # heading-rotated formation offsets
    FormW = opti.parameter(1, N_win)                   # phase-dependent formation weight
    EnfVYf = opti.parameter()                          # enforce terminal y-speed if 1 else off
    Tc0 = opti.parameter(N_uav)                        # previous-window tension (seam)

    # ── Objective ──
    cost = 0
    for k in range(N_win):
        e_track = posL[:, k] - Ref[:, k]
        cost += Q_track * ca.sqrt(ca.dot(e_track, e_track) + track_eps**2)

        for i in range(N_uav):
            e_form = pos[i][:, k] - (posL[:, k] + Off[i][:, k] + depth * e_z)
            cost += FormW[0, k] * Q_form * ca.dot(e_form, e_form)
            cost += R_T  * u[i][0, k]**2     # thrust effort
            cost += W_energy * u[i][0, k] * x[i][0, k]  # Power = T * V
            cost += R_a  * u[i][1, k]**2     # alpha
            cost += R_mu * u[i][2, k]**2     # bank
    e_term = posL[:, -1] - Ref[:, -1]
    cost += Q_term * ca.dot(e_term, e_term) + track_eps**2

    # Velocity smoothness: discourage large timestep-to-timestep airspeed jumps.
    for i in range(N_uav):
        cost += R_dV * ca.sumsqr(x[i][0, 1:] - x[i][0, :-1])

    # Tension regularization: min-norm split + smooth rate + window-seam continuity
    for i in range(N_uav):
        cost += R_Tc  * ca.sumsqr(Tc[i])                          # magnitude
        cost += R_dTc * ca.sumsqr(Tc[i][:, 1:] - Tc[i][:, :-1])   # rate
        cost += R_dTc * (Tc[i][:, 0] - Tc0[i])**2                 # seam to prev win
        cost += W_cab * ca.sumsqr(eps[i])                         # cable-band slack
    opti.minimize(cost)

    # ── Initial condition: full state fixed to the carried parameters ──
    # X0/pL0 already satisfy ||p_i - p_L|| = L_c by construction, so no separate
    # position-level cable equality is needed at node 0.
    opti.subject_to(posL[:, 0] == pL0)
    opti.subject_to(velL[:, 0] == vL0)
    for i in range(N_uav):
        opti.subject_to(x[i][:, 0] == X0[i])

    # ── Dynamics: integrate the whole coupled system one step at a time ──
    for k in range(N_win - 1):
        xs_next, pL_next, vL_next = coupled_step(
            [x[i][:, k] for i in range(N_uav)],
            [u[i][:, k] for i in range(N_uav)],
            [Tc[i][:, k] for i in range(N_uav)],
            posL[:, k], velL[:, k])
        for i in range(N_uav):
            opti.subject_to(x[i][:, k + 1] == xs_next[i])
        opti.subject_to(posL[:, k + 1] == pL_next)
        opti.subject_to(velL[:, k + 1] == vL_next)

    # ── Path constraints (state/control bounds, tension, rigid cable) ──
    for i in range(N_uav):
        for k in range(N_win):
            # State box bounds (keep away from V=0 and cos(gamma)=0 singularities)
            opti.subject_to(opti.bounded(V_min, x[i][0, k], V_max))      # airspeed
            opti.subject_to(opti.bounded(-gam_max, x[i][1, k], gam_max)) # gamma
            # Control box bounds
            opti.subject_to(opti.bounded(T_min,     u[i][0, k], T_max))
            opti.subject_to(opti.bounded(alpha_min, u[i][1, k], alpha_max))
            opti.subject_to(opti.bounded(-mu_max,   u[i][2, k], mu_max))
            # Cable tension non-negative and bounded (cables pull, never push)
            opti.subject_to(opti.bounded(0.0, Tc[i][:, k], Tc_max))

            # Cable length kept inside a SOFT band: double-sided inequality on the
            # squared chord length, relaxed by slack eps >= 0 (penalized in cost):
            #   (L-tol)^2 - eps <= ||d||^2 <= (L+tol)^2 + eps
            d    = pos[i][:, k] - posL[:, k]
            d_sq = ca.dot(d, d)
            opti.subject_to(eps[i][:, k] >= 0)
            opti.subject_to(d_sq >= (cable_len - cable_tol)**2 - eps[i][:, k])
            opti.subject_to(d_sq <= (cable_len + cable_tol)**2 + eps[i][:, k])

    # Collision avoidance: all pairs, all timesteps
    for i in range(N_uav):
        for j in range(i + 1, N_uav):
            for k in range(N_win):
                diff = pos[i][:, k] - pos[j][:, k]
                opti.subject_to(ca.dot(diff, diff) >= d_min**2)

    # Terminal payload y-speed requirement (enabled only on the true final window).
    vyL_f = velL[1, -1]
    opti.subject_to(vyL_f >= y_speed - vY_final_tol - (1 - EnfVYf) * vY_final_bigM)
    opti.subject_to(vyL_f <= y_speed + vY_final_tol + (1 - EnfVYf) * vY_final_bigM)

    opti.solver('ipopt', {'expand': True},
                {'max_iter': 1000, 'print_level': print_level,
                 'mu_strategy': 'adaptive', 'tol': 1e-6})
    return dict(opti=opti, x=x, u=u, Tc=Tc, posL=posL, velL=velL,
                X0=X0, pL0=pL0, vL0=vL0, Ref=Ref, Off=Off, FormW=FormW,
                EnfVYf=EnfVYf, Tc0=Tc0, N_win=N_win)


# ── Initial-guess helper (cold start over reference indices `idx`) ────────────

def set_guess(nlp, idx):
    """Hold the consistent cruise state along the reference slice `idx`."""
    opti = nlp['opti']
    Tc_guess = m_L * g * cable_len / (N_uav * depth)   # static cable share
    for i in range(N_uav):
        xg = np.zeros((6, len(idx)))
        for col, k in enumerate(idx):
            xg[0, col], xg[1, col], xg[2, col], xg[3:6, col] = uav_ref(k, i)
        opti.set_initial(nlp['x'][i], xg)
        ug = np.zeros((3, len(idx)))
        ug[0, :] = 5.0                 # thrust [N]
        ug[1, :] = np.deg2rad(2.0)     # alpha
        opti.set_initial(nlp['u'][i], ug)
        opti.set_initial(nlp['Tc'][i], Tc_guess * np.ones((1, len(idx))))
    opti.set_initial(nlp['posL'], r_ref[idx].T)
    opti.set_initial(nlp['velL'], vel_ref[idx].T)


def set_ic(nlp, xs, pL, vL):
    """Fix a window's carried initial-state parameters."""
    opti = nlp['opti']
    for i in range(N_uav):
        opti.set_value(nlp['X0'][i], xs[i])
    opti.set_value(nlp['pL0'], pL)
    opti.set_value(nlp['vL0'], vL)


def sol_arrays(sol, nlp):
    """Pull the solved window into plain numpy arrays."""
    return dict(
        x=[np.asarray(sol.value(nlp['x'][i])).reshape(6, nlp['N_win'])
           for i in range(N_uav)],
        u=[np.asarray(sol.value(nlp['u'][i])).reshape(3, nlp['N_win'])
           for i in range(N_uav)],
        Tc=[np.asarray(sol.value(nlp['Tc'][i])).reshape(1, nlp['N_win'])
            for i in range(N_uav)],
        posL=np.asarray(sol.value(nlp['posL'])).reshape(3, nlp['N_win']),
        velL=np.asarray(sol.value(nlp['velL'])).reshape(3, nlp['N_win']),
    )


# ── Solve: monolithic ─────────────────────────────────────────────────────────

def solve_monolithic():
    """One NLP over all N nodes (the original prototype, via the shared builder)."""
    nlp = build_nlp(N, print_level=5)
    set_ic(nlp, *consistent_ic())
    nlp['opti'].set_value(nlp['Ref'], r_ref.T)
    nlp['opti'].set_value(nlp['FormW'], form_w.reshape(1, -1))
    nlp['opti'].set_value(nlp['EnfVYf'], 1.0)
    for i in range(N_uav):
        nlp['opti'].set_value(nlp['Off'][i], offsets_rot[np.arange(N), i, :].T)
    # No previous window: anchor the seam term to the static cable share.
    Tc_guess = m_L * g * cable_len / (N_uav * depth)
    nlp['opti'].set_value(nlp['Tc0'], Tc_guess * np.ones(N_uav))
    set_guess(nlp, np.arange(N))
    t0 = time.time()
    sol = nlp['opti'].solve()
    print(f"\n[monolithic] solved in {time.time() - t0:.2f} s, "
          f"{nlp['opti'].stats()['iter_count']} IPOPT iters")
    return sol_arrays(sol, nlp)


# ── Solve: stitched receding horizon ──────────────────────────────────────────

def solve_rhc():
    """Slide a window of N_h nodes across the trajectory, committing N_apply nodes
    per solve and warm-starting from the shifted previous solution. The committed
    states are dynamically continuous because each window's IC is the previous
    window's first uncommitted state."""
    nlp  = build_nlp(N_h, print_level=0)
    opti = nlp['opti']

    X_full    = [np.zeros((6, N)) for _ in range(N_uav)]
    U_full    = [np.zeros((3, N)) for _ in range(N_uav)]
    Tc_full   = [np.zeros((1, N)) for _ in range(N_uav)]
    posL_full = np.zeros((3, N))
    velL_full = np.zeros((3, N))

    xs_cur, pL_cur, vL_cur = consistent_ic()
    Tc0_cur = (m_L * g * cable_len / (N_uav * depth)) * np.ones(N_uav)  # static share
    prev = None
    k, win = 0, 0
    t0 = time.time()

    while k < N - 1:
        # Reference slice; near the end, clamp by holding the final reference node.
        idx = np.minimum(np.arange(k, k + N_h), N - 1)
        set_ic(nlp, xs_cur, pL_cur, vL_cur)
        opti.set_value(nlp['Ref'], r_ref[idx].T)
        opti.set_value(nlp['FormW'], form_w[idx].reshape(1, -1))
        opti.set_value(nlp['EnfVYf'], float(idx[-1] == N - 1))
        for i in range(N_uav):
            opti.set_value(nlp['Off'][i], offsets_rot[idx, i, :].T)
        opti.set_value(nlp['Tc0'], Tc0_cur)

        # Warm start: shift previous solution forward by N_apply, else cold guess.
        if prev is None:
            set_guess(nlp, idx)
        else:
            for i in range(N_uav):
                xg = np.hstack([prev['x'][i][:, N_apply:],
                                np.tile(prev['x'][i][:, -1:], (1, N_apply))])
                xg[:, 0] = xs_cur[i]
                opti.set_initial(nlp['x'][i], xg)
                ug = np.hstack([prev['u'][i][:, N_apply:],
                                np.tile(prev['u'][i][:, -1:], (1, N_apply))])
                opti.set_initial(nlp['u'][i], ug)
                tg = np.hstack([prev['Tc'][i][:, N_apply:],
                                np.tile(prev['Tc'][i][:, -1:], (1, N_apply))])
                opti.set_initial(nlp['Tc'][i], tg)
            pg = np.hstack([prev['posL'][:, N_apply:],
                            np.tile(prev['posL'][:, -1:], (1, N_apply))])
            pg[:, 0] = pL_cur
            opti.set_initial(nlp['posL'], pg)
            vg = np.hstack([prev['velL'][:, N_apply:],
                            np.tile(prev['velL'][:, -1:], (1, N_apply))])
            vg[:, 0] = vL_cur
            opti.set_initial(nlp['velL'], vg)

        try:
            sol = opti.solve()
        except RuntimeError as err:
            # Warm start landed in a bad region: fall back to a cold guess once.
            print(f"[rhc] window {win} (nodes {k}..{k + N_h - 1}) warm-start "
                  f"solve failed ({err}); retrying from cold guess")
            set_guess(nlp, idx)
            for i in range(N_uav):
                opti.set_initial(nlp['x'][i][:, 0], xs_cur[i])
            opti.set_initial(nlp['posL'][:, 0], pL_cur)
            opti.set_initial(nlp['velL'][:, 0], vL_cur)
            sol = opti.solve()
        s = sol_arrays(sol, nlp)
        prev = s

        n_commit = min(N_apply, N - 1 - k)             # never commit the last node here
        for i in range(N_uav):
            X_full[i][:, k:k + n_commit]  = s['x'][i][:, :n_commit]
            U_full[i][:, k:k + n_commit]  = s['u'][i][:, :n_commit]
            Tc_full[i][:, k:k + n_commit] = s['Tc'][i][:, :n_commit]
        posL_full[:, k:k + n_commit] = s['posL'][:, :n_commit]
        velL_full[:, k:k + n_commit] = s['velL'][:, :n_commit]

        # Next window's IC is this window's first uncommitted node.
        xs_cur = [s['x'][i][:, n_commit] for i in range(N_uav)]
        pL_cur = s['posL'][:, n_commit]
        vL_cur = s['velL'][:, n_commit]
        Tc0_cur = np.array([s['Tc'][i][0, n_commit] for i in range(N_uav)])
        k   += n_commit
        win += 1

    # Final node
    for i in range(N_uav):
        X_full[i][:, N - 1]  = xs_cur[i]
        U_full[i][:, N - 1]  = U_full[i][:, N - 2]
        Tc_full[i][:, N - 1] = Tc_full[i][:, N - 2]
    posL_full[:, N - 1] = pL_cur
    velL_full[:, N - 1] = vL_cur
    print(f"\n[rhc] {win} window solves, {time.time() - t0:.2f} s total "
          f"(N_h={N_h}, N_apply={N_apply})")
    return dict(x=X_full, u=U_full, Tc=Tc_full, posL=posL_full, velL=velL_full)


# ── Run the chosen strategy ───────────────────────────────────────────────────

if SOLVE_MODE == 'monolithic':
    res = solve_monolithic()
elif SOLVE_MODE == 'rhc':
    res = solve_rhc()
else:
    raise ValueError(f"unknown SOLVE_MODE: {SOLVE_MODE!r}")

# ── Extract solution ──────────────────────────────────────────────────────────

pos_sol  = [res['x'][i][3:6, :] for i in range(N_uav)]   # list of (3, N)
V_sol    = [res['x'][i][0, :] for i in range(N_uav)]
T_sol    = [res['u'][i][0, :] for i in range(N_uav)]
alpha_sol = [res['u'][i][1, :] for i in range(N_uav)]
mu_sol   = [res['u'][i][2, :] for i in range(N_uav)]
Tc_sol   = [res['Tc'][i].flatten() for i in range(N_uav)]
posL_sol = res['posL']                                   # (3, N)
E_sol    = [np.sum(T_sol[i] * V_sol[i]) * dt for i in range(N_uav)]
E_total  = sum(E_sol)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# 3D trajectory
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(r_ref[:, 0], r_ref[:, 1], r_ref[:, 2],
         'k--', linewidth=2, label='Reference (payload)')
ax1.plot(posL_sol[0], posL_sol[1], posL_sol[2],
         'm-', linewidth=2, label='Payload')
for i in range(N_uav):
    c = colors[i % len(colors)]
    ax1.plot(pos_sol[i][0], pos_sol[i][1], pos_sol[i][2],
             color=c, linewidth=1.2, label=f'UAV {i+1}')
    ax1.scatter(*pos_sol[i][:, 0], color=c, marker='o', s=40)
    ax1.scatter(*pos_sol[i][:, -1], color=c, marker='*', s=90)
    ax1.plot([pos_sol[i][0, -1], posL_sol[0, -1]],
             [pos_sol[i][1, -1], posL_sol[1, -1]],
             [pos_sol[i][2, -1], posL_sol[2, -1]], color='gray', linewidth=0.8)
ax1.scatter(*posL_sol[:, -1], color='m', marker='s', s=60)
ax1.set_xlabel('north [m]')
ax1.set_ylabel('east [m]')
ax1.set_zlabel('h [m]')
ax1.set_title(f'UAV + payload trajectories ({SOLVE_MODE})')
ax1.legend(fontsize=8)

# Airspeed over time
ax2 = fig.add_subplot(132)
for i in range(N_uav):
    ax2.plot(t_vec, V_sol[i], color=colors[i % len(colors)], label=f'UAV {i+1}')
ax2.axhline(V_min, color='r', ls='--', lw=0.8)
ax2.axhline(V_max, color='r', ls='--', lw=0.8, label='V limits')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Airspeed V [m/s]')
ax2.set_title('Airspeed')
ax2.legend()
ax2.grid(True)

# Cable tensions over time
ax3 = fig.add_subplot(133)
for i in range(N_uav):
    ax3.plot(t_vec, Tc_sol[i], color=colors[i % len(colors)], label=f'cable {i+1}')
ax3.axhline(0.0, color='k', linewidth=0.8)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Cable tension [N]')
ax3.set_title('Cable tensions')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig('uav_formation_result.png', dpi=150)
plt.close()

# UAV maneuver histories
fig2, axs = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
turn_start_idx = np.searchsorted(s_ref, s_turn_start)
turn_end_idx = np.searchsorted(s_ref, s_turn_end)
turn_start_t = t_vec[min(turn_start_idx, N - 1)]
turn_end_t = t_vec[min(turn_end_idx, N - 1)]

for ax in axs:
    ax.axvspan(turn_start_t, turn_end_t, color='0.9', label='rounded turn')
    ax.grid(True)

for i in range(N_uav):
    c = colors[i % len(colors)]
    label = f'UAV {i+1}'
    axs[0].plot(t_vec, V_sol[i], color=c, label=label)
    axs[1].plot(t_vec, np.rad2deg(alpha_sol[i]), color=c, label=label)
    axs[2].plot(t_vec, T_sol[i], color=c, label=label)
    axs[3].plot(t_vec, np.rad2deg(mu_sol[i]), color=c, label=label)

axs[0].axhline(V_min, color='r', ls='--', lw=0.8)
axs[0].axhline(V_max, color='r', ls='--', lw=0.8)
axs[1].axhline(np.rad2deg(alpha_min), color='r', ls='--', lw=0.8)
axs[1].axhline(np.rad2deg(alpha_max), color='r', ls='--', lw=0.8)
axs[2].axhline(T_min, color='r', ls='--', lw=0.8)
axs[2].axhline(T_max, color='r', ls='--', lw=0.8)
axs[3].axhline(-np.rad2deg(mu_max), color='r', ls='--', lw=0.8)
axs[3].axhline(np.rad2deg(mu_max), color='r', ls='--', lw=0.8)
axs[0].set_ylabel('Airspeed [m/s]')
axs[1].set_ylabel('Angle of attack [deg]')
axs[2].set_ylabel('Thrust [N]')
axs[3].set_ylabel('Bank angle [deg]')
axs[3].set_xlabel('Time [s]')
axs[0].set_title('UAV maneuver histories')
axs[0].legend(loc='upper right')

plt.tight_layout()
plt.savefig('uav_maneuver_histories.png', dpi=150)
plt.close()

# ── Print summary ─────────────────────────────────────────────────────────────

print("\n=== Solution Summary ===")
print(f"Solve mode     : {SOLVE_MODE}")
print(f"Number of UAVs : {N_uav}    Timesteps: {N}    Euler substeps: {M_EULER}")
print(f"Payload mass   : {m_L} kg,  cable length: {cable_len} m")
print(f"Total thrust energy: {E_total:.2f} J")
print(f"Final payload tracking error: "
      f"{np.linalg.norm(posL_sol[:, -1] - r_ref[-1, :]):.4f} m")
for i in range(N_uav):
    print(f"UAV {i+1}: V in [{V_sol[i].min():.1f}, {V_sol[i].max():.1f}] m/s, "
          f"T max {T_sol[i].max():.2f} N, "
          f"energy {E_sol[i]:.2f} J, "
          f"|bank| max {np.rad2deg(np.abs(mu_sol[i])).max():.1f} deg, "
          f"cable T in [{Tc_sol[i].min():.1f}, {Tc_sol[i].max():.1f}] N")

# Check cable lengths (should equal cable_len)
print(f"\nCable length check (should equal {cable_len} m):")
for i in range(N_uav):
    lens = np.linalg.norm(pos_sol[i] - posL_sol, axis=0)
    print(f"  cable {i+1}: [{lens.min():.3f}, {lens.max():.3f}] m")

# Check minimum pairwise distances
print("\nMinimum pairwise distances:")
for i in range(N_uav):
    for j in range(i + 1, N_uav):
        dists = np.linalg.norm(pos_sol[i] - pos_sol[j], axis=0)
        print(f"  UAV {i+1} - UAV {j+1}: {dists.min():.3f} m  (limit: {d_min} m)")

err = np.linalg.norm(posL_sol - r_ref.T, axis=0)
print(f"\nPayload tracking error [m]: mean={err.mean():.3f}  max={err.max():.3f}  "
      f"final={err[-1]:.3f}")
