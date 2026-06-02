"""
Simple prototype: N point-mass UAVs tracking a reference trajectory together.

Each UAV is a 3D point mass controlled by a force vector.
Constraints:
  - Collision avoidance between all UAV pairs
  - Maximum force magnitude per UAV
  - Dynamics: simple double integrator

Objective:
  - Track a reference centroid trajectory
  - Maintain formation (UAVs stay close to their offset from centroid)
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Problem parameters ────────────────────────────────────────────────────────

N_uav   = 3          # number of UAVs (change freely)
N       = 100         # number of timesteps
dt      = 0.1        # timestep size [s]
m       = 1.0        # mass of each UAV [kg]

F_max   = 1.0       # max force magnitude per UAV [N]
d_min   = 1.0        # minimum distance between any two UAVs [m]

# Formation offsets: each UAV flies at a fixed offset from the centroid
# Evenly spaced on a horizontal circle of radius r_form
r_form  = 2.0
angles  = np.linspace(0, 2 * np.pi, N_uav, endpoint=False)
offsets = np.array([[r_form * np.cos(a), r_form * np.sin(a), 0.0]
                    for a in angles])  # shape (N_uav, 3)

# ── Reference centroid trajectory (simple helix) ─────────────────────────────

t_vec = np.linspace(0, (N - 1) * dt, N)
# r_ref = np.vstack([
#     2.0 * np.cos(0.5 * t_vec),   # x
#     2.0 * np.sin(0.5 * t_vec),   # y
#     0.5 * t_vec,                  # z (climbing)
# ]).T  # shape (N, 3)
r_ref = np.vstack([
    0.0 * t_vec,   # x
    0.0 * t_vec,   # y
    0.5 * t_vec,                  # z (climbing)
]).T  # shape (N, 3)



# ── CasADi Opti setup ─────────────────────────────────────────────────────────

opti = ca.Opti()

# Decision variables
# pos[i]: (3, N)  position of UAV i over time
# vel[i]: (3, N)  velocity of UAV i over time
# F[i]:   (3, N)  force    of UAV i over time

pos = [opti.variable(3, N) for _ in range(N_uav)]
# att = [opti.variable(3, N) for _ in range(N_uav)]
vel = [opti.variable(3, N) for _ in range(N_uav)]
F   = [opti.variable(3, N) for _ in range(N_uav)]

# ── Objective ─────────────────────────────────────────────────────────────────

cost = 0
Q_track   = 10.0   # weight: centroid tracking
Q_form    = 5.0    # weight: formation keeping
R_control = 0.1    # weight: control effort
CLIMB_ANGLE = np.pi/6 

for k in range(N):
    # Centroid of all UAVs at time k
    centroid = sum(pos[i][:, k] for i in range(N_uav)) / N_uav

    # Track reference centroid
    e_track = centroid - r_ref[k, :]
    cost += Q_track * ca.dot(e_track, e_track)

    # Each UAV should stay at its offset from centroid
    for i in range(N_uav):
        e_form = (pos[i][:, k]) - (r_ref[k, :])
        cost += Q_form * ca.dot(e_form, e_form)

    # Penalize control effort
    for i in range(N_uav):
        cost += R_control * ca.dot(F[i][:, k], F[i][:, k])

opti.minimize(cost)

# ── Constraints ───────────────────────────────────────────────────────────────

for i in range(N_uav):

    # --- Initial conditions: start at offset from first reference point
    p0 = r_ref[0, :] + offsets[i, :]
    opti.subject_to(pos[i][:, 0] == p0)
    opti.subject_to(vel[i][:, 0] == np.zeros(3))

    for k in range(N - 1):
        # Double integrator dynamics (Euler integration)
        opti.subject_to(
            pos[i][:, k + 1] == pos[i][:, k] + dt * vel[i][:, k]
        )
        opti.subject_to(
            vel[i][:, k + 1] == vel[i][:, k] + dt * (F[i][:, k] / m)
        )  
    
    # Max force constraint at each timestep
    for k in range(N):
        opti.subject_to(
            ca.dot(F[i][:, k], F[i][:, k]) <= F_max**2
        )
    
    # Max angle between vertical and horizontal plane
    for k in range(N):
        v = vel[i][:, k]

        horizontal_dist = ca.sqrt(v[0]**2 + v[1]**2 + 1e-6)
        angle = ca.atan2(v[2], horizontal_dist)

        opti.subject_to(
            angle <= CLIMB_ANGLE
        )

# Collision avoidance: all pairs, all timesteps
for i in range(N_uav):
    for j in range(i + 1, N_uav):
        for k in range(N):
            diff = pos[i][:, k] - pos[j][:, k]
            opti.subject_to(ca.dot(diff, diff) >= d_min**2)

# ── Initial guess: straight offset from reference ─────────────────────────────

for i in range(N_uav):
    #p_guess = (r_ref + offsets[i, :]).T   # shape (3, N)
    opti.set_initial(pos[i], np.zeros((3, N)))
    opti.set_initial(vel[i], np.zeros((3, N)))
    opti.set_initial(F[i],   np.zeros((3, N)))

# ── Solve ─────────────────────────────────────────────────────────────────────

opti.solver('ipopt', {}, {'max_iter': 1000, 'print_level': 3})
sol = opti.solve()

# ── Extract solution ──────────────────────────────────────────────────────────

pos_sol = [sol.value(pos[i]) for i in range(N_uav)]   # list of (3, N)
F_sol   = [sol.value(F[i])   for i in range(N_uav)]

centroid_sol = sum(pos_sol) / N_uav   # (3, N)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(12, 5))

# 3D trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(r_ref[:, 0], r_ref[:, 1], r_ref[:, 2],
         'k--', linewidth=2, label='Reference centroid')
ax1.plot(centroid_sol[0], centroid_sol[1], centroid_sol[2],
         'k-',  linewidth=1.5, label='Actual centroid')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i in range(N_uav):
    c = colors[i % len(colors)]
    ax1.plot(pos_sol[i][0], pos_sol[i][1], pos_sol[i][2],
             color=c, linewidth=1.5, label=f'UAV {i+1}')
    ax1.scatter(*pos_sol[i][:, 0], color=c, marker='o', s=50)
    ax1.scatter(*pos_sol[i][:, -1], color=c, marker='*', s=100)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_zlabel('z [m]')
ax1.set_title('UAV Trajectories')
ax1.legend(fontsize=8)

# Force magnitudes over time
ax2 = fig.add_subplot(122)
for i in range(N_uav):
    F_mag = np.linalg.norm(F_sol[i], axis=0)
    ax2.plot(t_vec, F_mag, color=colors[i % len(colors)], label=f'UAV {i+1}')
ax2.axhline(F_max, color='r', linestyle='--', label=f'F_max = {F_max} N')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Force magnitude [N]')
ax2.set_title('Control Forces')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('uav_formation_result.png', dpi=150)
plt.close()

# ── Print summary ─────────────────────────────────────────────────────────────

print("\n=== Solution Summary ===")
print(f"Number of UAVs : {N_uav}")
print(f"Timesteps      : {N}")
print(f"Final centroid error: "
      f"{np.linalg.norm(centroid_sol[:, -1] - r_ref[-1, :]):.4f} m")
for i in range(N_uav):
    print(f"UAV {i+1} max force: {np.linalg.norm(F_sol[i], axis=0).max():.2f} N")

# Check minimum pairwise distances
print("\nMinimum pairwise distances:")
for i in range(N_uav):
    for j in range(i + 1, N_uav):
        diffs = pos_sol[i] - pos_sol[j]
        dists = np.linalg.norm(diffs, axis=0)
        print(f"  UAV {i+1} - UAV {j+1}: {dists.min():.3f} m  (limit: {d_min} m)")
