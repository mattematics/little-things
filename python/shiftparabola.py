from pylab import *
import math

ax = subplot(111)
subplots_adjust(bottom=0.25)
t = arange(-10,10,0.001)
axis([-10,10,-10,10])

def parabola(scalex, scaley, delx, dely, x = t):
	return scaley*((scalex*(x + delx))**2 + dely)

graph, = plot(t, parabola(1,1,0,0), lw=2, color='blue')
origin = plot(0, 0, 'o', color='black')
title(r"Graph of $y = a \left ( \left ( b \left (x + \Delta x \right ) \right )^2 + \Delta y \right )$")
grid(True)

axscalex = axes([0.6,0.03,0.30,0.03])
axscaley = axes([0.6, 0.08, 0.30, 0.03])
axdelx = axes([0.6,0.13,0.30,0.03])
axdely = axes([0.6, 0.18, 0.30, 0.03])

sscalex = Slider(axscalex, r"scale x", -5, 5, valinit=1, valfmt='%1.2f')
sscaley = Slider(axscaley, r"scale y", -5, 5, valinit=1, valfmt="%1.2f")
sdelx = Slider(axdelx, r"$\Delta x$", -5, 5, valinit=0, valfmt='%1.2f')
sdely = Slider(axdely, r"$\Delta y$", -5, 5, valinit=0, valfmt="%1.2f")

def update(val):
	sx = sscalex.val
	sy = sscaley.val
	dx = sdelx.val
	dy = sdely.val
	
	graph.set_ydata(parabola(sx,sy,dx,dy))
		
	ax = subplot(111)
	title(r"Graph of $ y = " +\
		format(sy,'1.2f') +\
		r" \left ( \left (" +\
		format(sx,'1.2f') +\
		r"\left (" +\
		"x + " +\
		format(dx, '1.2f') +\
		r"\right ) \right )^2 + " +\
		format(dy, '1.2f') +\
		r"\right ) " +\
		r" $")
	
	draw()
	
sscalex.on_changed(update)
sscaley.on_changed(update)
sdelx.on_changed(update)
sdely.on_changed(update)

show()