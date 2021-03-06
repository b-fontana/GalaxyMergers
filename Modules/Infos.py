import time

class LoopInfo:
    """Handles all loop-related informations
    
    Arguments:
    1. total_len: number of items the loop will process'
    """
    def __init__(self, total_len=-1):
        self.total_len = total_len
        self.time_init = time.time()

    def loop_print(self, it, step_time):
        """
        The time is measured from the last loop_info call
        """
        subtraction = step_time - self.time_init
        if self.total_len == -1:
            print("{0:d} iteration. Iteration time: {1:.3f}" .format(it, subtraction))
        else:
            percentage = (float(it)/self.total_len)*100
            print("{0:.2f}% finished. Iteration time: {1:.3f}" .format(percentage, subtraction))
        self.time_init = time.time()
