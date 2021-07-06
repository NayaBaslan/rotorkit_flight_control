from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.integrate import solve_ivp

# Paramter obtained directly by measuring:

# Wind Speed
u = 5
rho = 1.225
int_steps = 500*2

# Blade radius
class Rotorblades():
    def __init_():
        self.n = 1
        self.m = 1
        self.R = 1
        self.name = 'blades'
    def compute_I(self):
         self.I = self.n*(1.0/3)*self.m*((self.R)**2)

short_blades = Rotorblades()
short_blades.r_0 = 0.07
short_blades.R = 0.225
short_blades.c = 0.026
short_blades.m = 0.0171
short_blades.n = 3
short_blades.compute_I()
short_blades.name = "R=220 (mm) model"

long_blades = Rotorblades()
long_blades.r_0 = 0.05
long_blades.R = 0.350
long_blades.c = 0.032
long_blades.m = 0.0252
long_blades.n = 3
long_blades.compute_I()
long_blades.name = "R=350 (mm) model"

#ui = (inflow velocity)

rad2deg = 180.0/np.pi
deg2rad = 1.0/rad2deg
rads2rpm = 30.0/np.pi
rpm2rads = 30.0/np.pi

#pitch angle
thetas_deg = [8.0, 10.0, 14.0]




def gen_fun(theta, b=long_blades):
    omega = SX.sym('omega')
    # radius to integrate over
    r = SX.sym('r')
    # flow angle

    lambda_r = r*omega/u
    a= 1.0/3
    ap = a*(1-a)/lambda_r**2

    phi = np.arctan(u/(omega*r))
    #up = u + ui
    # angle of attack
    alpha = phi - theta


    # lift coefficient
    #  c_l = - 2.56460052e+01*alpha**3 - 4.43234451e-02*alpha**2 + 7.00607112*alpha + 1.02584291e-03
    #  c_l = 7*alpha
    c_l = 1.5 * np.tanh(5.5*alpha)
    # lift force
    dF_l = 0.5*rho*b.c*(u**2+(omega*r)**2)*c_l

    # drag coefficient
    c_d = 1.84470128e-02*alpha**3 + 4.39400675e-01*alpha**2-3.71962200e-04*alpha + 5.74006568e-03
    #  c_d = 9.40977430e+01*alpha**6 -1.82128683e-01*alpha**5 - 8.20854743*alpha**4 + 1.84470128e-02*alpha**3 + 4.39400675e-01*alpha**2-3.71962200e-04*alpha + 5.74006568e-03

    # drag force
    dF_d = 0.5*rho*b.c*(u**2+(omega*r)**2)*c_d

    # tangential torque
    d_tau_sym = (np.sin(phi)*dF_l - np.cos(phi)*dF_d)*r
    d_thrust_sym = (np.cos(phi)*dF_l + np.sin(phi)*dF_d)
    # generate integrator
    d_tau_f = Function('f', [r], [d_tau_sym])
    d_thrust_f = Function('f', [r], [d_thrust_sym])

    lower_bound = SX.sym('lower_bound')
    upper_bound = SX.sym('upper_bound')
    params = [lower_bound, upper_bound]
    h = (upper_bound-lower_bound)/int_steps

    #total torque
    integral = 0
    for i in range(int_steps):
        integral = integral + d_tau_f(lower_bound+i*h)
    integral = integral * h

    integral_thrust = 0
    for i in range(int_steps):
        integral_thrust = integral_thrust + d_thrust_f(lower_bound+i*h)
    integral = integral * h

    return Function('integral_fun', [vertcat(omega), vertcat(*params)], [vertcat(integral)])

def gen_odefun(blades, theta):
    torque = gen_fun(theta, b=blades)

    def ode(t,omega):
        return torque(omega,[blades.r_0,blades.R])/blades.I

    return ode

if __name__ == "__main__":
    t_0 = 0
    t_end = 30

    plt.figure()
    blades = short_blades
    plt.title(f"{blades.name}")
    for theta in thetas_deg:
        sol = solve_ivp(gen_odefun(blades, theta*deg2rad), [t_0, t_end],[0], t_eval=np.linspace(t_0,t_end,100))
        plt.plot(sol.t,sol.y[0]*rads2rpm, label=f"pitch={theta} deg")

    plt.legend()
    plt.grid()
    plt.savefig(f"./figs/{blades.name.replace(' ','_')}.png")

    plt.figure()
    blades = long_blades
    plt.title(f"{blades.name}")


    for theta in thetas_deg:
        sol = solve_ivp(gen_odefun(blades, theta*deg2rad), [t_0, t_end],[0], t_eval=np.linspace(t_0,t_end,100))
        plt.plot(sol.t,sol.y[0]*rads2rpm, label=f"pitch={theta} deg")

    plt.legend()
    plt.grid()
    plt.savefig(f"./figs/{blades.name.replace(' ','_')}.png")
