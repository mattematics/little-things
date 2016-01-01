from pylab import *
import math

# demo showing how a probability distribution updates with more information
# according to bayes rule
# 2 sets of controls:
# first one lets you set the # of heads out of a # of tosses
#   (the heads and tosses sliders)
# second one lets you advance through steps of a precomputed sequence of tosses
#   (the simulation step slider)

ax = subplot(111)
subplots_adjust(bottom=0.25)
t = arange(0,1,0.001)
axis([0,1,0,1.1])

def prior(x = t):
    return array([1 for tt in t])

def likelihood(heads, tosses, x = t):
    o = prior() * (((x)**(heads))*((1-x)**(tosses-heads)))
    o /= max(o)
    return o

graph, = plot(t, likelihood(0,0), lw=2, color='blue')
origin = plot(0, 0, 'o', color='black')
title(r"Graph of $y=f(x)$")
grid(True)

axheads = axes([0.6, 0.18, 0.30, 0.03])
axtosses = axes([0.6,0.13,0.30,0.03])
axsim = axes([0.6,0.05,0.30,0.03])

sheads = Slider(axheads, r"Heads", 0, 1000, valinit=0, valfmt="%1.2f")
stosses = Slider(axtosses, r"Tosses", 0, 1000, valinit=0, valfmt='%1.2f')
ssim = Slider(axsim, r"Simulation Step", 0, 1000, valinit=0, valfmt="%1.2f")

def update(val):
    if(stosses.val < sheads.val):
        stosses.set_val(sheads.val)
    
    heads = sheads.val
    tosses = stosses.val

    graph.set_ydata(likelihood(heads,tosses))

    ax = subplot(111)
    title(r"Graph of $y=f(x)$")
    
    draw()

sheads.on_changed(update)
stosses.on_changed(update)

weight = random()
def simulate(n = 1000, w = weight):
    heads = 0
    data = [0]
    for k in xrange(n):
        if(random() < w):
            heads += 1
        data.append(heads)
    return(data)

data = simulate()

def update_sim(val):
    heads = data[int(val)]
    tosses = int(val)
    
    graph.set_ydata(likelihood(heads, tosses))

    ax = subplot(111)
    title(r"Posterior with $"+str(heads)+r"$ heads out of $"+str(tosses)+r"$ tosses."+"\nWeighting = "+ format(str(weight), '1.5'))
    
    draw()
    
ssim.on_changed(update_sim)

show()