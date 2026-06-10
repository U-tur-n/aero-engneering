"""
A Stand-Alone Module for Testing Lifting-Line Theory for Tapered Wings
Technical contact: maruyama.daigo@nihon-u.ac.jp
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt

"""
The following website by Geoffrey Nyaga was referenced and modified for proper numerical computations to solve linear systems in general (Ax=b):
https://github.com/geoffreynyaga/lifting-line-theory

For a further understanding of the lifting-line theory, you can refer to:
http://fnorio.com/0117man_powered_aeroplane0/man_powered_aeroplane0.html

See also the textbook, Section 3.10 and Subsection 3.10.2.
"""


### PROBLEM SETTING (for a tapered wing) ###
"""
We do not need to define "the free stream velocity V [m/s]" to compute CL, Cd, Cl(y), and Cd(y) (though we need to define it to compute the circulation distribution Γ(y)).
"""
S = 122.6                                                     # wing area [m2]
AR = 10.47 #15.                                                   # aspect ratio [-]
taper = 0.27 #0.75                                              # taper ratio [-]
twist_angle = -1.8 #-2.                                           # twist angle at the wing tip [deg.] as the difference

AoA = 1.                                                        # angle of attack (AoA), wing setting angle (and the fixed angle at the wing root [deg.])
AoA_0 = -4.2                                                    # zero-lift angle of attack [deg.]
sigma = 0.85                                                    # 0.8-0.9 [-] (for normal airfoil)

N = 9                                                           # number of segments [-] (for lifting-line theory)
#############################################

print("\n----------------------------------------------")
print("AR                    =", AR)
print("taper                 =", taper)
print("twist_angle           =", twist_angle)
print("angle of attack (AoA) =", AoA)
print("----------------------------------------------\n")


# --- the other parameters accordingly obtained by the problem setting ---
m_inf = 2.*np.pi*sigma                                          # lift curve slope [1/rad] (See the textbook p.137.)
b = np.sqrt(AR*S)                                               # wing span [m]
c_root = 2.*S / ( b*(1.+taper) )                                # chord of wing root [m]
alpha = np.linspace(AoA, AoA+twist_angle, N)                    # list: alpha (angle of attack) distribution [deg.] in the spanwise direction y
#print("alpha distribution in the span direction y from wing root to wing tip [deg.] =\n", alpha)

theta = np.linspace( np.pi/2., np.pi/(2*N), N)                  # list: theta [rad], 0 < theta <= 0.5pi (the range of theta used to represent coordinate y). theta=0 is excluded because it is a singularity and its solution is trivial.
y = 0.5*b*np.cos(theta)                                         # list: y[m], the coordinate in the span direction (where y is represented by theta)
c = c_root * ( 1. - ( 1. - taper ) * np.cos(theta) )            # list: chord distribution c(y) [m] in the spanwise direction

mu = c*m_inf / (4.*b)                                           # a parameter appearing in the linear system below
# ------------------------------------------------------------------------


"""
Solving Prandtl's integral equation (Eq. (3.105b) in the textbook) to obtain the circulation distribution Γ(y) yields the following linear system.
"""
# --- solving linear system (x for Ax=d) ----
d = mu * (alpha - AoA_0) / (180./np.pi)                                               # note: "alpha - AoA_0" is called "absolute angle of attack" (See the textbook p.137).
A = np.array( [ np.sin(i*theta) * ( 1. + i*mu / np.sin(theta) ) for i in range(1, 2*N+1, 2) ] ).T
luA = scipy.linalg.lu_factor(A)                                                         # Using LU decomposition to avoid numerical errors in direct computation of inverse matrices
x = scipy.linalg.lu_solve(luA, d)                                                       # the solution of the linear system
#print("x =", x)
# -------------------------------------------




# --- circulation distribution Γ(y) ---
"""
Circulation distribution:
Γ(y) = Γ(y; V, b, x, theta)
requiring the definition of the free stream velocity V
"""
# -------------------------------------


# --- 3D Lift (CL) --------------------
CL = np.pi*AR*x[0]                                                                      # Eq. (3.112) in the textbook
print("CL         =", CL)
# -------------------------------------


# --- 3D Induced Drag (CD_induced) ----
delta = sum( (2*i+1) * np.power(x[i], 2) for i in range(1,N) ) / np.power(x[0], 2)
CD_induced = np.power(CL, 2) / (np.pi*AR) * (1.+delta)                                  # Eq. (3.114) in the textbook
print("CD_induced =", CD_induced)
# ------------------------------------


# --- 2D lift distribution Cl(y) -----
"""
Cl(y) = rho*V*Γ(y)              by Kutta–Joukowski theorem
      = ...
      = Cl(y; without Γ(y))     calculated without Γ(y) eventually
"""
Cl = 4.*b / c * sum( (np.sin( (2*i+1)*theta) ) * x[i] for i in range(N) )               # no integral in Eq. (3.111) in the textbook; then Eq. (3.112) eliminates rho and V by substituting c=S/b.

Cl = np.append(Cl, 0.)          # adding Cl=0 (trivial) at the singularity point (the wing tip)
y = np.append(y, 0.5*b)         # adding y=0.5b as it is the y-coordinate of the wing tip

plt.plot(y, Cl, marker="o")
plt.title("Cl distribution")
plt.xlabel("semi-span location: y [m]")
plt.ylabel("lift coefficient: Cl [-]")
#plt.grid()
plt.savefig("Cl_distribution_AR{}_taper{}_twist{}_AoA{}.jpg".format(AR, taper, twist_angle, AoA))
# mgr = plt.get_current_fig_manager()
# mgr.window.wm_geometry("+1920+0")
plt.show()
# ------------------------------------
print("")

###############
