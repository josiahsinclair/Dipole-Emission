import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy import linalg
from matplotlib.widgets import Slider  # import the Slider widget

import time as time

starttime = time.time()
mpl.rcParams.update({'font.size': 10, 'font.family': 'STIXGeneral',
                            'mathtext.fontset': 'stix'})




#constants
k = 2*2*np.pi
wavelength = 2*np.pi/k
print("Wavelength is",wavelength)


def E_rad(x,y,shift): #in regular coordinates
	r_0 = np.sqrt(x**2+y**2)
	theta_0 = np.arccos(x/r_0)
	r_1 = np.sqrt(r_0**2 + shift**2 - 2*r_0*shift*np.cos(theta_0))
	theta_1 = np.arctan(y/(np.abs(x-shift)))
	A = np.sin(theta_1)*np.exp(1j*k*r_1)/(r_1+1)
	return A





# atom_locations = np.arange(-1,1.1,.5)
spacing = 1.0

X = np.arange(-10.00, 10.00, .05)
Y = np.arange(-2.00, 2.00, .05)
# plt.figure()
# plt.plot(Y,abs(E_rad(0.001,Y,0)))

x, y = np.meshgrid(X, Y)


print(np.max(abs(E_rad(x,y,0))))
print(np.max(abs(E_rad(x,y,17))))

def TotalElectricField(spacing,x,y):
	atom_locations = np.arange(-5*spacing/2.0,5*spacing/2.0,spacing/2.0)
	B = 0*x + 0*1j*x
	for i in range(len(atom_locations)):
		B += E_rad(x, y,atom_locations[i])#*np.exp(1j*k*r**2 + shift**2)
	return B









fig = plt.figure(figsize=(8,6))
func_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
plt.axes(func_ax) # select sin_ax
atom_locations_init = np.arange(-5*spacing/2.0,5*spacing/2.0,spacing/2.0)

funcplot = plt.contourf(x,y,abs(TotalElectricField(spacing,x,y)),50, alpha = 1)
# atom_locations_init = np.array([-5*spacing/2.0,-4*spacing/2.0, -3*spacing/2, -2*spacing/2.0,-1*spacing/2.0,0,spacing/2.0,2*spacing/2.0,3*spacing/2.0,4*spacing/2.0,5*spacing/2.0])
funcplot2, = plt.plot(atom_locations_init,0*atom_locations_init,'ro', color = 'white')



# here we create the slider
a_min = .2    # the minimial value of the paramater a
a_max = 5   # the maximal value of the paramater a
a_init = .5   # the value of the parameter a to be used initially, when the graph is created

a_slider = Slider(slider_ax,      # the axes object containing the slider
                  'a',            # the name of the slider parameter
                  a_min,          # minimal value of the parameter
                  a_max,          # maximal value of the parameter
                  valinit=a_init  # initial value of the parameter
                 )


# Next we define a function that will be executed each time the value
# indicated by the slider changes. The variable of this function will
# be assigned the value of the slider.
def update(a):
    plt.contourf(x,y,abs(TotalElectricField(a,x,y)),50, alpha = 1) # set new y-coordinates of the plotted points
    funcplot2.set_xdata(np.array(np.arange(-5*a/2.0,5*a/2.0,a/2.0))) # set new y-coordinates of the plotted points
    fig.canvas.draw_idle()          # redraw the plot

# the final step is to specify that the slider needs to
# execute the above function when its value changes
a_slider.on_changed(update)

# plt.plot(x_values,y_values, color = 'red')
# plt.arrow(-15,0,10,0,color = 'red', head_width = .2, head_length = 1)
# plt.arrow(-5,0,-10,0,color = 'red', head_width = .2, head_length = 1)
# plt.plot(atom_locations,0*atom_locations,'ro', color = 'white') 
# plt.text(-18,.2,'polarization direction', color = 'white', fontsize = 15)

stoptime = time.time()
print("Program took %1.2f seconds" %(stoptime-starttime))
plt.show()
