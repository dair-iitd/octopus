from scipy.stats import poisson

class PoissonProcess(object):

    def __init__(self,arrivalRatePerMinute):
        self.arrivalRatePerMinute = arrivalRatePerMinute

    def generateArrivals(self,timePeriodInMinutes):
        return poisson.rvs(timePeriodInMinutes*self.arrivalRatePerMinute)

    