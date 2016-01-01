from pylab import *
import math

ax = subplot(111)
subplots_adjust(bottom=0.25)
t = arange(-10,10,0.001)
axis([-10,10,-10,10])

def trajectory(b, u):
	b = radians(b)
	t = arange(-10,10,0.001)
	return t*tan(b) - t*t*5/(u*u*cos(b)*cos(b))
	
def ground(a):
	a = radians(a)
	t = arange(-10,10,0.001)
	return tan(a)*t
	
def intersect(a, b, u):
	a, b = radians(a), radians(b)
	return -0.2*u*u*(1/cos(a))*cos(b)*sin(a-b)

line, = plot(t, ground(0), lw=2, color='red')
path, = plot(t, trajectory(45,10), lw=2, color='blue')
maxpath, = plot(t, trajectory(0.5*0 + 45, 10), lw=2, color='yellow')
origin = plot(0, 0, 'o', color='black')
pt = intersect(0,45,10)
point, = plot(pt, tan(radians(0))*pt, 's', color='green')
title(r"$"+str(pt)+r"$ units until intersect")
grid(True)

axa = axes([0.6,0.05,0.30,0.03])
axb = axes([0.6, 0.10, 0.30, 0.03])
axu = axes([0.6,0.15,0.30, 0.03])

sa = Slider(axa, "ground angle", -45, 45, valinit=0)
sb = Slider(axb, "launch angle", 0, 180, valinit=45)
su = Slider(axu, "speed", 0, 20, valinit=10)

def update(val):
	a = sa.val
	b = sb.val
	u = su.val
	
	path.set_ydata(trajectory(b,u))
	line.set_ydata(ground(a))
	maxpath.set_ydata(trajectory(0.5*a + 45, u))
	pt = intersect(a,b,u)
	point.set_xdata(pt)
	point.set_ydata(tan(radians(a))*pt)
	
	ax = subplot(111)
	title(r"$"+str(pt)+r"$ units until intersect")
	
	draw()
	
sa.on_changed(update)
sb.on_changed(update)
su.on_changed(update)

show()