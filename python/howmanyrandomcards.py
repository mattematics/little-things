#how many random cards do you need to see to complete a deck?

from pylab import *
from random import *

def test(N=10**6):
    # returns how many random card selections it takes
    # to obtain a full deck of cards
    
    # represents a dect of cards as 52 bits
    # and uses bit operations for tests

    # starting empy deck is 0...0
    my_deck = 0b0
    # a full deck is 1...1
    full_deck = 2**52 - 1

    # N times, we pick a random card and add it to our deck
    # N is just a bound in case we're very unlucky
    for i in xrange(N):
        found = 2**int(random()*52)
        # if we haven't seen this card
        if my_deck & found == 0:
            # add it to the deck
            my_deck += found
        # if the deck is full
        if my_deck ^ full_deck == 0:
            return i
            break

num_experiments = 10000

# prepare a list that we will populate with
# the results of test()
scores = zeros(num_experiments, dtype='int')

for i in xrange(num_experiments):
    # add a result of test to our list
    scores[i] = test()
    
largest_required = max(scores)
least_required = min(scores)
average = mean(scores)

print "in ", num_experiments, " trials..."
print "avg needed: ", average
print "most needed: ", largest_required
print "least needed: ", least_required
print median(scores)

# make a histogram
n, bins, patches = hist(scores, 100, normed=1)
show()