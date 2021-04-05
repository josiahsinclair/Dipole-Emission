import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy import linalg


import time as time

starttime = time.time()
mpl.rcParams.update({'font.size': 10, 'font.family': 'STIXGeneral',
                            'mathtext.fontset': 'stix'})




#constants
k = 1
epsilon_0 = 1
alpha = 1
E_in = 1


# def E_rad(r,theta): #in polar coordinates
# 	A = k**2*np.sin(theta)*np.exp(1j*k*r)*alpha*E_in/(4*3.14159*epsilon_0*r)
# 	return A



# def E_rad(x,y,shift): #in regular coordinates
# 	r_0 = np.sqrt(x**2+y**2)
# 	if x>0:
# 		theta_0 = np.arcsin(y/r_0)
# 	if x<=0:
# 		theta_0 = np.pi - np.arcsin(y/r_0)
# 	r_1 = np.sqrt(r_0**2 + shift**2 - 2*r_0*shift*np.cos(theta_0))
# 	theta_1 = np.arctan(y/(np.abs(x-shift)))
# 	A = k**2.0*np.sin(theta_1)*np.exp(1j*k*r_1)*alpha*E_in/(4*3.14159*epsilon_0*r_1)
# 	return A


def E_rad2(x,y,shift): #in regular coordinates
	r_0 = np.sqrt(x**2+y**2)
	theta_0 = np.arccos(x/r_0)
	r_1 = np.sqrt(r_0**2 + shift**2 - 2*r_0*shift*np.cos(theta_0))
	theta_1 = np.arctan(y/(np.abs(x-shift)))
	A = np.sin(theta_1)*np.exp(1j*k*r_1)/(r_1+1)
	return A

# # print(E_rad(x,y,x0))
# print(E_rad2(.02,.02,0))
# print(E_rad2(.02-.01,.02,-.01))

# print(E_rad2(.02,.02,0))
# print(E_rad2(.02+.01,.02,+.01))


# print(E_rad2(-.02,.02,0))
# print(E_rad2(-.02+.01,.02,+.01))



# print(E_rad2(-.02,.02,0))
# print(E_rad2(-.02+.03,.02,+.03))


# print(E_rad2(-.01,-.001,0))
# print(E_rad2(-.01+.2,-.001,+.2))
# print("try something else")
# print(E_rad2(.19,.0001,+.2))
# print(E_rad2(.21,.0001,+.2))




X = np.arange(-1.01, 1.01, .02)
Y = np.arange(-1.01, 1.01, .02)
# plt.figure()
# plt.plot(X,np.real(E_rad2(X,.001,0)))
# plt.plot(X, np.real(E_rad2(X,.001,.2)))


x, y = np.meshgrid(X, Y)




E_test1 = np.abs(np.real(E_rad2(x,y,0)))
print(np.max(E_test1))

plt.figure(figsize = (9,6))
plt.contourf(x,y,E_test1,50, alpha = 1)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.title("abs(E_rad)")
plt.colorbar()
plt.arrow(.4,0,.2,0,color = 'red')
plt.arrow(.6,0,-.2,0,color = 'red')
plt.plot(0,0,'ro', color = 'white') 
plt.text(.18,.05,'polarization direction', color = 'white', fontsize = 15)
# plt.ylim(-.5,.5)
# plt.xlim(-.5,.5)
stoptime = time.time()
print("Program took %1.2f seconds" %(stoptime-starttime))
plt.show()
