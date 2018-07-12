#Checks if a list is empty, i.e., if all the nested lists inside have zero elements
def isListEmpty(inList):
    if isinstance(inList, list): # it is a list
        return all( map(isListEmpty, inList) ) # all([]) is True
    return False # it is not a list   

def var_range(start,stop,stepiter):
    step = iter(stepiter)
    while start < stop:
        yield start
        start += next(step)
