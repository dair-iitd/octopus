import numpy as np
import os

'''
Set up the experiment using this class. This class contains (mostly) all meta-data
about the experiment. Objects of this class are passed to other classes to set up their
parameters appropriately.
'''

class SystemParameters(object):

    numQuestions = 0
    path = '/Users/krandiash/Downloads/octopus/'
    #UPDATE path to point to where the octopus folder is located (which contains this file)

    def __init__(self,
                 difficultyDistribution,
                 workerDistribution,
                 value=200,
                 timeGranularityInMinutes=10,
                 numQuestions=100,
                 ballotsToCompletionGranularity=10,
                 completenessGranularity=10,
                 numberOfPricePoints=4,
                 deadline=60,
                 timeReward=100000,
                 costRewardScalingFactor=1.0,
                 costMDPDiscountFactor=1.0,
                 bvFunctionGranularity=40,
                 synchronizationGranularity=1,
                 controllerType=1,
                 numberOfSimulations=100,
                 workerArrivalRates=(),
                 numWorkers=1000,
                 threshold=0.9):

        self.numQuestions = numQuestions

        self.difficultyDistribution = difficultyDistribution
        self.workerDistribution = workerDistribution

        self.currentPrice = 1 #Price increases in increments of 1 unit (1 -> 2 ...).
        self.value = value #Change self.value to get changes in effective price.

        self.averageGamma = workerDistribution.getMean()

        self.timeGranularityInMinutes = timeGranularityInMinutes
        self.ballotsToCompletionGranularity = ballotsToCompletionGranularity
        self.completenessGranularity = completenessGranularity #Define in number of divisions
        self.numberOfPricePoints = numberOfPricePoints

        self.deadline = deadline
        self.timeReward = timeReward
        self.costRewardScalingFactor = costRewardScalingFactor
        self.costMDPDiscountFactor = costMDPDiscountFactor
        self.bvFunctionGranularity = bvFunctionGranularity
        self.synchronizationGranularity = synchronizationGranularity
        self.controllerType = controllerType
        self.numberOfSimulations = numberOfSimulations
        for x in workerArrivalRates:
            for y in x:
                x[y] = x[y] * timeGranularityInMinutes
        self.workerArrivalRates = workerArrivalRates#np.array(workerArrivalRates) * timeGranularityInMinutes
        self.numWorkers = numWorkers
        self.threshold = threshold

    def setWorkerArrivalRates(self,workerArrivalRates):
        for x in workerArrivalRates:
            for y in x:
                x[y] = x[y] * self.timeGranularityInMinutes
        self.workerArrivalRates = workerArrivalRates

    '''
    Use this to generate question IDs when adding a question (and nowhere else).
    '''
    @staticmethod
    def addQuestion():
        SystemParameters.numQuestions += 1
        return SystemParameters.numQuestions - 1

    def prettyPrint(self):
        print "----------------------------"
        print "System Parameters"
        print "----------------------------"
        print "Difficulty Distribution: Beta(%2.2f,%2.2f)" % (self.difficultyDistribution.alpha,self.difficultyDistribution.beta)
        print "Worker Distribution: Gamma(%2.2f,%2.2f)" % (self.workerDistribution.k,self.workerDistribution.theta)
        print "Penalty for Wrong Answer: %d" % (self.value)
        print "Time Granularity in Minutes: %d" % (self.timeGranularityInMinutes)
        print "Number of Questions: %d" % (self.numQuestions)
        print "Ballots to Completion Granularity: %d" % (self.ballotsToCompletionGranularity)
        print "Completeness Granularity: %d" % (self.completenessGranularity)
        print "Number of Price Points: %d" % (self.numberOfPricePoints)
        print "Deadline: %d" % (self.deadline)
        print "Time Reward: %d" % (self.timeReward)
        print "Cost Reward Scaling Factor: %1.3f" % (self.costRewardScalingFactor)
        print "Cost MDP Discount Factor: %.4f" % (self.costMDPDiscountFactor)
        print "BV Function Granularity: %d" % (self.bvFunctionGranularity)
        print "Synchronization Granularity: %d" % (self.synchronizationGranularity)
        print "Controller Type: %d" % (self.controllerType)
        print "Number of Simulations: %d" % (self.numberOfSimulations)
        print "Worker Arrival Rates: " + str([x[0] for x in self.workerArrivalRates])

    def prettyReturn(self):
        return "----------------------------\n"\
               + "System Parameters\n"\
               + "----------------------------\n"\
               + "Difficulty Distribution: Beta(%2.2f,%2.2f)\n" % (self.difficultyDistribution.alpha,self.difficultyDistribution.beta)\
               + "Worker Distribution: Gamma(%2.2f,%2.2f)\n" % (self.workerDistribution.k,self.workerDistribution.theta)\
               + "Penalty for Wrong Answer: %d\n" % (self.value)\
               + "Time Granularity in Minutes: %d\n" % (self.timeGranularityInMinutes)\
               + "Number of Questions: %d\n" % (self.numQuestions)\
               + "Ballots to Completion Granularity: %d\n" % (self.ballotsToCompletionGranularity)\
               + "Completeness Granularity: %d\n" % (self.completenessGranularity)\
               + "Number of Price Points: %d\n" % (self.numberOfPricePoints)\
               + "Deadline: %d\n" % (self.deadline)\
               + "Time Reward: %d\n" % (self.timeReward)\
               + "Cost Reward Scaling Factor: %1.3f\n" % (self.costRewardScalingFactor)\
               + "Cost MDP Discount Factor: %.4f\n" % (self.costMDPDiscountFactor)\
               + "BV Function Granularity: %d\n" % (self.bvFunctionGranularity)\
               + "Synchronization Granularity: %d\n" % (self.synchronizationGranularity)\
               + "Controller Type: %d\n" % (self.controllerType)\
               + "Number of Simulations: %d\n" % (self.numberOfSimulations)\
               + "Worker Arrival Rates: " + str([x[0] for x in self.workerArrivalRates]) + "\n"



    def stringify(self):
        l = [self.difficultyDistribution.alpha,self.difficultyDistribution.beta,self.workerDistribution.k,self.workerDistribution.theta,
                self.value,self.timeGranularityInMinutes,self.numQuestions,self.ballotsToCompletionGranularity,self.completenessGranularity,
                self.numberOfPricePoints, self.deadline, self.timeReward, self.costRewardScalingFactor, self.costMDPDiscountFactor, self.bvFunctionGranularity,
                self.controllerType,self.synchronizationGranularity]
        l.extend([x[0] for x in self.workerArrivalRates])
        l = [str(x) for x in l]
        return ",".join(l)  + "_" + str(self.numberOfSimulations)

    def stringifyForBVFunction(self,referenceDifficultyDistribution):
        l = [self.difficultyDistribution.alpha,self.difficultyDistribution.beta,self.workerDistribution.k,self.workerDistribution.theta,
                self.value,self.numberOfPricePoints,self.bvFunctionGranularity,referenceDifficultyDistribution.alpha,referenceDifficultyDistribution.beta]
        l = [str(x) for x in l]
        return ",".join(l)

    def stringifyForTransitionTable(self):
        l = [self.difficultyDistribution.alpha,self.difficultyDistribution.beta,self.workerDistribution.k,self.workerDistribution.theta,
                self.value,self.numQuestions,self.ballotsToCompletionGranularity,self.completenessGranularity,
                self.numberOfPricePoints, self.bvFunctionGranularity,self.controllerType]
        l = [str(x) for x in l]
        return ",".join(l)

    def stringifyQuestionData(self):
        l = [self.difficultyDistribution.alpha,self.difficultyDistribution.beta,self.numQuestions]
        l = [str(x) for x in l]
        return ",".join(l)

    def stringifyWorkerData(self):
        l = [self.workerDistribution.k,self.workerDistribution.theta,self.numWorkers]
        l = [str(x) for x in l]
        return ",".join(l)

    def stringifyArrivalData(self,cost):
        return ",".join([str(x) for x in [cost,self.workerArrivalRates[cost - 1][0]]])
