import numpy as np

def unison_shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def isListEmpty(inList):
    if isinstance(inList, list): # it is a list 
        return all( map(isListEmpty, inList) ) # all([]) is True
    return False # it is not a list

def var_range(start,stop,stepiter):
    step = iter(stepiter)
    while start < stop:
        yield start
        start += next(step)
