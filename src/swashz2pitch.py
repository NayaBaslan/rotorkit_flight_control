import numpy as np
import matplotlib.pyplot as plt

rad2deg = 180/np.pi

def collective(z, debug=0):
    l3 = 22
    l2 = 42
    l1 = 15
    l4 = 42

    z = l4 - z
    z_p = np.sqrt(l1**2+z**2)
    if debug: print(z_p)
    if debug: print(l1/z_p)
    theta_p = np.arctan(l1/z)
    if debug: print(theta_p)
    # arccos theorem
    cos_theta_pp = (l2**2-l3**2-l1**2-z**2)/(2*l3*z_p)
    if debug: print(cos_theta_pp)
    theta_pp = np.arccos(cos_theta_pp)
    return (theta_p+theta_pp)*rad2deg -90


plt.figure()
zs = np.linspace(0, 25, 25);
plt.plot(zs, [collective(zi) for zi in zs], label="pitch_angle")
plt.grid()
plt.show()
