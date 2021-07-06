# For multiple pitch angles

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.integrate import solve_ivp

# Paramter obtained directly by measuring:

R = 0.35  # Blade radius
u = 4 # Wind Speed
rho = 1.225
c = 0.026
int_steps = 500*2
bs = [0.0, 5, 10, 15] #pitch angle
r_0 = 0.028 #initial radius element
#ui = (inflow velocity)
# beta
thetas = [ 15, 17, 20, 24] #pitch angle
omega = SX.sym('omega')
#acceptable range: 13 to 
theta = 20
theta_rad = theta/(2*np.pi)


for k in thetas:
   theta = k
   theta_rad = k/(2*np.pi)
   def gen_fun(omega):
        # radius to integrate over
        r = SX.sym('r')
        # flow angle
        phi = np.arctan(u/(omega*r))
        #up = u + ui
        # angle of attack
        alpha = phi - theta_rad

        # lift coefficient
        c_l = - 2.56460052e+01*alpha**3 - 4.43234451e-02*alpha**2 + 7.00607112*alpha + 1.02584291e-03
        # lift force
        dF_l = 0.5*rho*c*(u**2+(omega*r)**2)*c_l
        #dF_l = 0.5*rho*c*(up**2+(omega*r)**2)*c_l

        # drag coefficient
        c_d = 9.40977430e+01*alpha**6 -1.82128683e-01*alpha**5 - 8.20854743*alpha**4 + 1.84470128e-02*alpha**3 + 4.39400675e-01*alpha**2-3.71962200e-04*alpha + 5.74006568e-03
        # drag force
        dF_d = 0.5*rho*c*(u**2+(omega*r)**2)*c_d
        #dF_d = 0.5*rho*c*(up**2+(omega*r)**2)*c_d
        # tangential torque 

        f_sym = (np.sin(phi)*dF_l - np.cos(phi)*dF_d)*r
        # generate integrator
        f = Function('f', [r], [f_sym])
        lower_bound = SX.sym('lower_bound')
        upper_bound = SX.sym('upper_bound')
        params = [lower_bound, upper_bound]
        h = (upper_bound-lower_bound)/int_steps
        integral = 0
        for i in range(int_steps):
            integral = integral + f(lower_bound+i*h)
        integral = integral * h #total force

        Force_Total = Function('integral_fun', [vertcat(omega), vertcat(*params)], [vertcat(integral)])

        return Force_Total

   F = gen_fun(omega)


   #Torque_Total = F(0,[r_0,R])*R
   #print(Torque_Total)

   def ode_rotorkite(t,y):
       return F(y,[r_0,R])*R

   sol = solve_ivp(ode_rotorkite, [0, 800],[0.1], t_eval=np.linspace(0,800,100)) # initial value problem
   for i in range(0,len(sol.y)):
       plt.plot(sol.t,sol.y[i,:]/(np.pi)*180*0.4,label=theta)

   print(sol.y)
   print(theta)
   print('END_')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Rotational Velocity (rad/sec)')
plt.show()

