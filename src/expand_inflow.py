# This file models both cyclic and collective pitch control does the calculation over 3 blades
# adding inflow velocity component

# For multiple pitch angles

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.integrate import solve_ivp

# Paramter obtained directly by measuring:

R = 0.2  # Blade radius
u = 4 # Wind Speed

uwind = 4
u1 = 4
u2 = 3.5
ui = 0

rho = 1.225
c = 0.026
int_steps = 500*2
bs = [0.0, 5, 10, 15] #pitch angle
r_0 = 0.028 #initial radius element
#ui = (inflow velocity)
# beta
collective_pitch = 16#pitch angle
omega = SX.sym('omega')
#acceptable range: 13 to 
cyclic_lat = 20
cyclic_lon = 20

# azimuth angle is the integral of rotational velocity
#w = dgamma/dt
#gamma = integral(w)dt

m = 0.090


gamma = SX.sym('gamma') 
r = SX.sym('r')

thetas = [ 15, 17, 20, 24]


for k in thetas:
   collective_pitch=k
   def gen_fun(omega,gamma,ui):
        a = (u1-u2)/u1
        ct = 4*a*(1-a)
        ui = -0.5*(1- np.sqrt(1- ct))
        #print(a)
        u = uwind + ui
        print(ui)
        theta1 = collective_pitch + cyclic_lat*np.cos(gamma) + cyclic_lon*np.sin(gamma)
        theta2 = collective_pitch + cyclic_lat*np.cos(gamma+ 2*np.pi/3) + cyclic_lon*np.sin(gamma+ 2*np.pi/3)
        theta3 = collective_pitch + cyclic_lat*np.cos(gamma+ 4*np.pi/3) + cyclic_lon*np.sin(gamma + 4*np.pi/3)

        theta_rad1 = theta1/(180)*np.pi
        theta_rad2 = theta2/(180)*np.pi
        theta_rad3 = theta3/(180)*np.pi

        phi = np.arctan(u/(omega*r))
        alpha1 =  -theta_rad1  + phi
        alpha2 =  -theta_rad2 + phi
        alpha3 =  -theta_rad3  + phi
        
        # lift coefficient
        c_l1 = 1.5 * np.tanh(5.5*alpha1)
        c_l2 = 1.5 * np.tanh(5.5*alpha2)
        c_l3 = 1.5 * np.tanh(5.5*alpha3)
        
        # lift force
        dF_l1 = 0.5*rho*c*(u**2+(omega*r)**2)*c_l1
        dF_l2 = 0.5*rho*c*(u**2+(omega*r)**2)*c_l2
        dF_l3 = 0.5*rho*c*(u**2+(omega*r)**2)*c_l3

        # drag coefficient
        c_d1 = 1.84470128e-02*alpha1**3 + 4.39400675e-01*alpha1**2-3.71962200e-04*alpha1 + 5.74006568e-03
        c_d2 = 1.84470128e-02*alpha2**3 + 4.39400675e-01*alpha2**2-3.71962200e-04*alpha2 + 5.74006568e-03
        c_d3 = 1.84470128e-02*alpha3**3 + 4.39400675e-01*alpha3**2-3.71962200e-04*alpha3 + 5.74006568e-03

        # drag force
        dF_d1 = 0.5*rho*c*(u**2+(omega*r)**2)*c_d1
        dF_d2 = 0.5*rho*c*(u**2+(omega*r)**2)*c_d2
        dF_d3 = 0.5*rho*c*(u**2+(omega*r)**2)*c_d3
        
        # tangential torque 

        f_sym = (np.sin(phi)*(dF_l1+dF_l2+dF_l3) - np.cos(phi)*(dF_d1+dF_d2+dF_d3))*r

        d_tau_sym = (np.sin(phi)*(dF_l1+dF_l2+dF_l3) - np.cos(phi)*(dF_d1+dF_d2+dF_d3))*r
        d_tau_f = Function('f', [r], [d_tau_sym])
        # generate integrator
        f = Function('f', [r], [f_sym])
        lower_bound = SX.sym('lower_bound')
        upper_bound = SX.sym('upper_bound')
        params = [lower_bound, upper_bound]
        h = (upper_bound-lower_bound)/int_steps
        integral = 0
        for i in range(int_steps):
            integral = integral + d_tau_f(lower_bound+i*h)
        integral = integral * h #total force

        Force_Total = Function('integral_fun', [vertcat(omega),vertcat(gamma), vertcat(*params)], [vertcat(integral)])
        ui = -0.5*(1- np.sqrt(1- ct))
        return Function('integral_fun', [vertcat(omega), vertcat(gamma),vertcat(*params)], [vertcat(integral)])

   F = gen_fun(omega,gamma,ui)


   def ode_rotorkite(t,y):
        return F(y,t*y ,[r_0,R])

#        return F(y,t*y ,[r_0,R])/(m*R**2)

   sol = solve_ivp(ode_rotorkite, [0, 90000],[0.1],t_eval=np.linspace(0,7000,100)) # initial value problem

   gamma_angle = np.zeros(sol.t.size)
   for q in range(sol.t.size):
      gamma_angle[q] = sol.t[q]*sol.y[0,q]


   print('END_')

   gamma_angle = ( gamma_angle + np.pi) % (2 * np.pi ) - np.pi

   print(sol.t[0])
   for i in range(0,len(sol.y)):
       plt.plot(sol.t,sol.y[i,:]/(np.pi)*180*0.2)

   plt.xlabel('Time (s)')
   plt.ylabel('Rotational Velocity (rad/s)')


plt.show()

