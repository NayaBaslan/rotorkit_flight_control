from casadi import *
import numpy as np
import matplotlib.pyplot as plt
#import rk4_fun_def

def rk4_fun_def(omega, dynamics,steps, params):
    k1 = dynamics(omega,params)
    k2 = dynamics(omega+h/2*k1,params)
    k3 = dynamics(omega+h/2*k2,params)
    k4 = dynamics(omega+h*k3,params)
    x_next = omega + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x_next


# Define constants
R = 0.2
u = 4
rho = 1.225
c = 0.026
int_steps = 5000
bs = [0.0, 5, 10, 15]
r_0 = 0.028
b = 5
m = 0.2
L = 0.3
omega = SX.sym('omega')
rk4_steps = 5

# radius to integrate over
r = SX.sym('r')
# flow angle
phi = np.arctan(u/(omega*r))
# angle of attack
alpha = phi - b
# lift coefficient
c_l = - 2.56460052e+01*alpha**3 - 4.43234451e-02*alpha**2 + 7.00607112*alpha + 1.02584291e-03
# lift force
dF_l = 0.5*rho*c*(u**2+(omega*r)**2)*c_l
# drag coefficient
c_d = 9.40977430e+01*alpha**6 -1.82128683e-01*alpha**5 - 8.20854743*alpha**4 + 1.84470128e-02*alpha**3 + 4.39400675e-01*alpha**2-3.71962200e-04*alpha + 5.74006568e-03
# drag force
dF_d = 0.5*rho*c*(u**2+(omega*r)**2)*c_d
# tangential torque
f_sym = (np.sin(phi)*dF_l - np.cos(phi)*dF_d)*r
# generate integrator
f = Function('f', [r], [f_sym]) # torque function
lower_bound = SX.sym('lower_bound')
upper_bound = SX.sym('upper_bound')
params = [lower_bound, upper_bound]
h = (upper_bound-lower_bound)/int_steps
integral = 0.0
for i in range(int_steps):
    integral = integral + f(lower_bound+i*h)
integral = integral * h

integral_fun = Function('integral_fun', [vertcat(omega), vertcat(*params)], [vertcat(integral)])
torque_equation = Function('integral_fun', [vertcat(omega), vertcat(*params)], [vertcat(integral)])
w = 0 
radius = 0.028
print(torque_equation(w,[0.028,0.2]))
#torque_equation(w,[0.028,0.2]
I_total = 3.0*(1/3)*m*(L**2)
integral_omega = integral/I_total
w_dot = Function('integral_fun', [vertcat(omega), vertcat(*params)], [vertcat(integral_omega)])
print(w_dot(0,[0.028,0.2]))
xk = MX.sym('xk', 1,1)

for i in range(rk4_steps):
    xk = rk4_fun_def(omega,w_dot , int_steps/rk4_steps , vertcat(*params))
#Now use rk4 to discretize this ode
rk4_fun = Function('rk4_fun',[vertcat(omega), vertcat(*params)],{xk})

#DONE: find ode of angular acceleration in terms of torque and omega
