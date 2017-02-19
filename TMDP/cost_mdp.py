from scipy.stats import poisson, beta
import mdptoolbox.mdp
import numpy as np
from scipy.sparse import csr_matrix
import sys
sys.path.insert(0, '../Controller')
from controller import *
from greedy import *
from fractional import *
from simple import *
from hybrid import *
sys.path.insert(0, '../QualityPOMDP')
from quality_pomdp import *
sys.path.insert(0, '../')
from question import *
from difficulty_distribution import *
from worker_distribution import *
from system_parameters import *
from sklearn.preprocessing import normalize
import cPickle as pickle
import os
from copy import deepcopy
import matplotlib.pyplot as plt


from scipy.special import betainc


class CostMDP(object):

    def __init__(self,systemParameters):
        self.systemParameters = systemParameters

        self.possibleV_bar =  [round(x,2) for x in np.linspace(0, 1.0, self.systemParameters.completenessGranularity + 1)]
        self.possibleCosts = np.linspace(self.systemParameters.currentPrice,self.systemParameters.currentPrice + self.systemParameters.numberOfPricePoints - 1, self.systemParameters.numberOfPricePoints)

        self.adjustedArrivalRates = self.systemParameters.workerArrivalRates

        self.BVFunction = self.determineBVFunction(self.systemParameters.difficultyDistribution)

        self.maxBallots = max(self.BVFunction[1])

        self.maxBallotsDict = {x:max(self.BVFunction[x]) for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        self.overallMaxBallotsDict = {x:int((self.systemParameters.numQuestions * max(self.BVFunction[x])) + self.systemParameters.ballotsToCompletionGranularity - ((self.systemParameters.numQuestions * max(self.BVFunction[x])) % self.systemParameters.ballotsToCompletionGranularity)) for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        self.possibleE_b = {x:np.linspace(0, self.overallMaxBallotsDict[x], self.overallMaxBallotsDict[x]/float(self.systemParameters.ballotsToCompletionGranularity) + 1) for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}

        self.validE_b = self.determineAdmissibleBallotsToCompletion()

        self.numStates = 0

        self.stateSpaceMapping = []
        self.inverseStateSpaceMapping = {}
        self.numActions = 4

        self.validSameCostTransitions, self.validCrossCostTransitions = self.createValidStateTransitionTableUpdated()

        self.transitionProbabilityTable = self.createTransitionProbabilityTable()

        self.T = None
        self.R = None
        self.policy = None

        self.learnPolicy()

        self.currentState = None
        self.resetState()

    def resetState(self):
        self.currentState = self.inverseStateSpaceMapping[(max(self.validE_b[1][0]), 0.0, 0.0, self.systemParameters.currentPrice)]

    '''
    We use this function to compute the v_bar v/s E_b curves for different costs. Note that we are fixing a reference
    difficulty distribution to facilitate this calculation, as well as (implicitly) reference worker distribution.
    These curves are treated as piece-wise constant functions.
    Use bvFunctionGranularity to set the number of intervals for the piecewise constant functions. Default = 40 intervals.
    '''
    def determineBVFunction(self,referenceDifficultyDistribution):
        os.chdir(SystemParameters.path + 'TMDP')
        if os.path.exists('log/bvFunctions/%s' % (self.systemParameters.stringifyForBVFunction(referenceDifficultyDistribution))):
            print "Reading BV function."
            bvFunction = pickle.load(open('log/bvFunctions/%s' % (self.systemParameters.stringifyForBVFunction(referenceDifficultyDistribution))))
            os.chdir(SystemParameters.path)
            if not os.path.exists('Plots/BVFunction/%s.eps' % (self.systemParameters.stringifyForBVFunction(referenceDifficultyDistribution))):
                fig, ax = plt.subplots()
                ax.xaxis.set_ticks_position('none') 
                ax.yaxis.set_ticks_position('none') 
                plt.rc('font', family='serif')
                fig.set_size_inches(3.5, 2.0)
                fig.tight_layout()
                for cost,func in bvFunction.items():
                    plt.plot(2*np.linspace(0.5,1.0,self.systemParameters.bvFunctionGranularity + 1) - 1,func,label='$c = %d$' % (cost),marker='o',linewidth=0.5,ms=3)
                plt.xlabel("$v'_i$",fontsize=12)
                plt.ylabel("$\hat{B}_i$",fontsize=12)
                plt.legend(loc='best',prop={'size':10})
                fontsize = 8
                ax = plt.gca()

                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                plt.savefig('Plots/BVFunction/%s.eps' % (self.systemParameters.stringifyForBVFunction(referenceDifficultyDistribution)), format='eps', dpi=1000,bbox_inches='tight')

            return bvFunction

        print "Determining BV function."
        os.chdir(SystemParameters.path + 'QualityPOMDP')
        bvFunction = {}
        for cost in self.possibleCosts:
            bvFunction[cost] = []
            qualityPOMDPPolicy = QualityPOMDPPolicy(self.systemParameters,self.systemParameters.value,cost)
            qualityPOMDP = QualityPOMDP(None,qualityPOMDPPolicy)
            for v in np.linspace(0.5,1.0,self.systemParameters.bvFunctionGranularity + 1):
                qualityPOMDP.qualityPOMDPBelief.setBelief(v,referenceDifficultyDistribution.alpha,referenceDifficultyDistribution.beta)
                ballots = qualityPOMDP.estimateBallotsToCompletionUsingFrontierFinding(8)
                bvFunction[cost].append(ballots)
        os.chdir(SystemParameters.path + 'TMDP')
        print "Writing BV function."
        pickle.dump(bvFunction,open('log/bvFunctions/%s' % (self.systemParameters.stringifyForBVFunction(referenceDifficultyDistribution)),'w'))
        return bvFunction

    def estimateExpectedBallotsToCompletion(self,n,v_bar,piecewiseConstantFunction,intervals):
        estimate = piecewiseConstantFunction[-1]
        for i in xrange(len(intervals)):
            estimate += (piecewiseConstantFunction[i] - piecewiseConstantFunction[i+1]) * betainc(n*v_bar,n*(1 - v_bar), intervals[i])
        return estimate


    def reconstructDistribution(self,E_b,v_bar,cost):
        intervals = 2*np.linspace(0.5 + 0.5/self.systemParameters.bvFunctionGranularity,1.0,self.systemParameters.bvFunctionGranularity) - 1
        piecewiseConstantFunction = self.BVFunction[cost]

        # if E_b >= self.overallMaxBallotsDict[cost] - self.systemParameters.numQuestions:
        #     estimate,n_chosen = self.biModal(intervals,piecewiseConstantFunction,v_bar,E_b)
        #     if not abs(estimate - E_b) > self.systemParameters.ballotsToCompletionGranularity * 0.5:
        #         return n_chosen * v_bar, n_chosen * (1 - v_bar)
        #     other_estimate,other_n_chosen = self.nonBiModal(intervals,piecewiseConstantFunction,v_bar,E_b)
        #     if abs(estimate - E_b) < abs(other_estimate - E_b):
        #         return n_chosen * v_bar, n_chosen * (1 - v_bar)
        #     else:
        #         return other_n_chosen * v_bar, other_n_chosen * (1 - v_bar)
        # else:
        other_estimate,other_n_chosen = self.nonBiModal(intervals,piecewiseConstantFunction,v_bar,E_b)
        if not abs(other_estimate - E_b) > self.systemParameters.ballotsToCompletionGranularity * 0.5:
            return other_n_chosen * v_bar, other_n_chosen * (1 - v_bar)
        estimate,n_chosen = self.biModal(intervals,piecewiseConstantFunction,v_bar,E_b)
        if abs(estimate - E_b) < abs(other_estimate - E_b):
            return n_chosen * v_bar, n_chosen * (1 - v_bar)
        else:
            return other_n_chosen * v_bar, other_n_chosen * (1 - v_bar)

        # n_chosen = -1
        # error = 100000000000
        # v_min = min(v_bar,1-v_bar)
        # #Non Bi-modal distributions only
        # for n in np.linspace(1/v_min,100.0,400):
        #     estimate = self.estimateExpectedBallotsToCompletion(n,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
        #     if abs(estimate - E_b) < error:
        #         n_chosen = n
        #         error = abs(estimate - E_b)
        # estimate = self.estimateExpectedBallotsToCompletion(n_chosen,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
        # print estimate,E_b
        # if not abs(estimate - E_b) > self.systemParameters.ballotsToCompletionGranularity * 0.5:
        #     return n_chosen * v_bar, n_chosen * (1 - v_bar)
        # for n in np.linspace(0.01,1/v_min,200):
        #     estimate = self.estimateExpectedBallotsToCompletion(n,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
        #     if abs(estimate - E_b) < error:
        #         n_chosen = n
        #         error = abs(estimate - E_b)
        # estimate = self.estimateExpectedBallotsToCompletion(n_chosen,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
        # print estimate,E_b
        # return n_chosen * v_bar, n_chosen * (1 - v_bar)

    def biModal(self,intervals,piecewiseConstantFunction,v_bar,E_b):
        n_chosen = -1
        error = 100000000000
        v_min = min(v_bar,1-v_bar)
        for n in np.linspace(0.01,1/v_min,200):
            estimate = self.estimateExpectedBallotsToCompletion(n,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
            if abs(estimate - E_b) < error:
                n_chosen = n
                error = abs(estimate - E_b)
        estimate = self.estimateExpectedBallotsToCompletion(n_chosen,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
        return estimate, n_chosen

    def nonBiModal(self,intervals,piecewiseConstantFunction,v_bar,E_b):
        n_chosen = -1
        error = 100000000000
        v_min = min(v_bar,1-v_bar)
        for n in np.linspace(1/v_min,100.0,400):
            estimate = self.estimateExpectedBallotsToCompletion(n,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
            if abs(estimate - E_b) < error:
                n_chosen = n
                error = abs(estimate - E_b)
        estimate = self.estimateExpectedBallotsToCompletion(n_chosen,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
        return estimate, n_chosen

    '''
    Every E_b value that is achievable (from 0 to max) needs to have an associated state. Additionally, these values
    are achievable for every single cost. This function returns a cost-indexed dictionary, which lays out the (v_bar,E_b)
    pairs that are possible for each cost.
    '''
    def determineAdmissibleBallotsToCompletion(self):
        #TODO: Every v_bar should be achievable and a higher EB should not occur without a lower EB for a higher v_bar
        os.chdir(SystemParameters.path + 'TMDP')
        if os.path.exists('log/admissibleStates/%s' % (self.systemParameters.stringifyForTransitionTable())):
            print "Reading admissible states."
            validE_b = pickle.load(open('log/admissibleStates/%s' % (self.systemParameters.stringifyForTransitionTable())))
            with open('admissibleStates', 'w') as f:
                for key,value in validE_b.items():
                    f.write(str(key) + '\t\t' + str(value) + '\n')
            return validE_b

        print "Determining admissible states."
        intervals = 2*np.linspace(0.5 + 0.5/self.systemParameters.bvFunctionGranularity,1.0,self.systemParameters.bvFunctionGranularity) - 1
        tol = self.systemParameters.ballotsToCompletionGranularity * 0.5
        boundedIntervals = {x:[] for x in self.possibleCosts}
        validE_b = {x:[set() for _ in self.possibleV_bar] for x in self.possibleCosts}
        for cost in self.possibleCosts:
            print 'Cost: %d' % (cost)
            piecewiseConstantFunction = self.BVFunction[cost]

            for v_bar in self.possibleV_bar:
                print '\t%.2f' % (v_bar)
                if v_bar == 0.0:
                    boundedIntervals[cost].append((self.overallMaxBallotsDict[cost],self.overallMaxBallotsDict[cost]))
                    continue
                elif v_bar == 1.0:
                    boundedIntervals[cost].append((0,0))
                    continue
                lowerBound = 100000000000
                upperBound = -1
                #v_min = min(v_bar,1-v_bar)
                for n in np.linspace(0.01,50.0,500):
                    estimate = self.estimateExpectedBallotsToCompletion(n,v_bar,piecewiseConstantFunction,intervals) * self.systemParameters.numQuestions
                    if estimate < lowerBound:
                        lowerBound = estimate
                    if estimate > upperBound:
                        upperBound = estimate
                if upperBound - lowerBound < self.systemParameters.ballotsToCompletionGranularity:
                    delta = self.systemParameters.ballotsToCompletionGranularity - (upperBound - lowerBound)
                    lowerBound -= delta
                    if lowerBound < 0:
                        upperBound += delta

                boundedIntervals[cost].append((lowerBound-tol,upperBound+tol))

            for E_b in self.possibleE_b[cost]:
                covered = False
                closest_i = -1
                smallestDifference = 10000000
                for i,(lowerBound,upperBound) in enumerate(boundedIntervals[cost]):
                    if E_b == 0 and lowerBound <= self.systemParameters.ballotsToCompletionGranularity:
                        validE_b[cost][i].add(E_b)
                        covered = True
                        continue

                    if lowerBound <= E_b and E_b <= upperBound:
                        #validE_b[cost][i].add(E_b)
                        for otherCost in self.possibleCosts:
                            E_b_to_add = E_b - (self.overallMaxBallotsDict[cost] - self.overallMaxBallotsDict[otherCost])
                            if E_b_to_add <= 0:
                                E_b_to_add = 0
                            validE_b[otherCost][i].add(E_b_to_add)
                        covered = True
                    else:
                        if abs(lowerBound - E_b) < abs(upperBound - E_b) and abs(lowerBound - E_b) < smallestDifference:
                            smallestDifference = abs(lowerBound - E_b)
                            closest_i = i
                        elif abs(lowerBound - E_b) >= abs(upperBound - E_b) and abs(upperBound - E_b) < smallestDifference:
                            smallestDifference = abs(upperBound - E_b)
                            closest_i = i

                if not covered:
                    #validE_b[cost][closest_i].add(E_b)
                    for otherCost in self.possibleCosts:
                        E_b_to_add = E_b - (self.overallMaxBallotsDict[cost] - self.overallMaxBallotsDict[otherCost])
                        if E_b_to_add <= 0:
                            E_b_to_add = 0
                        validE_b[otherCost][closest_i].add(E_b_to_add)

        print "Writing admissible states."
        pickle.dump(validE_b,open('log/admissibleStates/%s' % (self.systemParameters.stringifyForTransitionTable()),'w'))
        with open('admissibleStates', 'w') as f:
            for key,value in validE_b.items():
                f.write(str(key) + '\t\t' + str(value) + '\n')
        return validE_b


    def learnPolicy(self):
        print "Creating state space mapping."
        self.createStateSpaceMapping()
        print "Creating reward and transition functions."
        self.createRewardAndTransitionFunction()
        print "Learning policy."
        finiteHorizonMDP = mdptoolbox.mdp.ValueIteration(self.T,self.R,self.systemParameters.costMDPDiscountFactor,max_iter=1000000)
        finiteHorizonMDP.verbose = True
        finiteHorizonMDP.run()

        self.policy = finiteHorizonMDP.policy


    def createTransitionProbabilityTable(self):
        print "Finding probabilities for transition."
        transitionProbabilityTable = {cost:{time:({},{}) for time in np.linspace(0, self.systemParameters.deadline, self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes + 1)} for cost in self.possibleCosts}
        for cost in self.possibleCosts:
            E_b_max = self.overallMaxBallotsDict[cost]
            for time in np.linspace(0, self.systemParameters.deadline, self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes + 1):
                for numBallots in self.possibleE_b[cost]:
                    if numBallots != E_b_max:
                        transProb = poisson._cdf(numBallots + self.systemParameters.ballotsToCompletionGranularity*0.5, self.adjustedArrivalRates[int(cost)-1][time]) \
                                - np.nan_to_num(float(poisson._cdf(numBallots - self.systemParameters.ballotsToCompletionGranularity*0.5, self.adjustedArrivalRates[int(cost)-1][time])))
                    else:
                        transProb = poisson._cdf(1000*E_b_max + self.systemParameters.ballotsToCompletionGranularity*0.5, self.adjustedArrivalRates[int(cost)-1][time]) \
                                - np.nan_to_num(float(poisson._cdf(E_b_max - self.systemParameters.ballotsToCompletionGranularity*0.5, self.adjustedArrivalRates[int(cost)-1][time])))
                    transitionProbabilityTable[cost][time][0][numBallots] = transProb
                sum = 0
                for E_b_limit in reversed(self.possibleE_b[cost]):
                    sum += transitionProbabilityTable[cost][time][0][E_b_limit]
                    transitionProbabilityTable[cost][time][1][E_b_limit] = sum

        return transitionProbabilityTable

    def findBestStatePair(self,v_bar,E_b,cost):
        v_bar_final = 0
        min_distance = 100000000
        for v in self.possibleV_bar:
            if abs(v - v_bar) <= min_distance:
                min_distance = abs(v - v_bar)
                v_bar_final = v

        closest_E_b = -10000000
        for possible_E_b in self.validE_b[cost][int(round(v_bar_final * self.systemParameters.completenessGranularity))]:
            if abs(possible_E_b - E_b) < abs(closest_E_b - E_b):
                closest_E_b = possible_E_b

        if closest_E_b < 0:
            if E_b % self.systemParameters.ballotsToCompletionGranularity >= self.systemParameters.ballotsToCompletionGranularity*0.5:
                E_b_final = E_b - (E_b % self.systemParameters.ballotsToCompletionGranularity)
                E_b_final = int(E_b_final + self.systemParameters.ballotsToCompletionGranularity)
            else:
                E_b_final = E_b - (E_b % self.systemParameters.ballotsToCompletionGranularity)

            closest_v_bar = -1
            if E_b_final not in self.validE_b[cost][int(round(v_bar_final * self.systemParameters.completenessGranularity))]:
                for potential_v_bar in self.possibleV_bar:
                    if E_b_final in self.validE_b[cost][int(round(potential_v_bar * self.systemParameters.completenessGranularity))]:
                        if abs(v_bar_final - closest_v_bar) > abs(v_bar_final - potential_v_bar):
                            closest_v_bar = potential_v_bar
                v_bar_final = closest_v_bar
        else:
            E_b_final = closest_E_b

        return v_bar_final, E_b_final

    def adjustVBar(self,v_bar,E_b,cost):
        closest_v_bar = -1
        v_bar_final = v_bar
        if E_b not in self.validE_b[cost][int(round(v_bar_final * self.systemParameters.completenessGranularity))]:
            for potential_v_bar in self.possibleV_bar:
                if E_b in self.validE_b[cost][int(round(potential_v_bar * self.systemParameters.completenessGranularity))]:
                    if abs(v_bar_final - closest_v_bar) > abs(v_bar_final - potential_v_bar):
                        closest_v_bar = potential_v_bar
            v_bar_final = closest_v_bar

        return v_bar_final, E_b


    def getBestAction(self):
        return self.policy[self.currentState]

    def executeActionUpdated(self,action,numBallots=0):
        if self.currentState == self.numStates - 1:
            return
        E_b1, tau, v_bar1, c_1 = self.stateSpaceMapping[self.currentState]
        numBallotsUpdated = numBallots - numBallots % self.systemParameters.ballotsToCompletionGranularity
        if numBallots % self.systemParameters.ballotsToCompletionGranularity > 0.5 * self.systemParameters.ballotsToCompletionGranularity:
            numBallotsUpdated = numBallotsUpdated + self.systemParameters.ballotsToCompletionGranularity
        if numBallotsUpdated > E_b1:
            numBallotsUpdated = E_b1
        if action == 0:
            E_b2, v_bar2 = self.validSameCostTransitions[(E_b1,v_bar1,c_1,numBallotsUpdated)]
            self.currentState = self.inverseStateSpaceMapping[(E_b2, tau + self.systemParameters.timeGranularityInMinutes, v_bar2, c_1)]
        elif action == 1:
            E_b2 = self.validCrossCostTransitions[(E_b1,v_bar1,c_1,c_1+1)]
            self.currentState = self.inverseStateSpaceMapping[(E_b2, tau, v_bar1, c_1+1)]
        elif action == 2:
            E_b2 = self.validCrossCostTransitions[(E_b1,v_bar1,c_1,c_1-1)]
            self.currentState = self.inverseStateSpaceMapping[(E_b2, tau, v_bar1, c_1-1)]
        else:
            self.currentState = self.numStates - 1


    def synchronizeState(self, qualityPOMDPs):
        E_b1, tau, v_bar1, c_1 = self.stateSpaceMapping[self.currentState]
        c_2 = c_1
        v_bar2 = reduce(lambda x,y:x + y,map(lambda x: 2*x.qualityPOMDPBelief.v_max - 1,qualityPOMDPs)) / float(self.systemParameters.numQuestions)

        # Comment for partial sync
        buckets = {}
        for qualityPOMDP in qualityPOMDPs:
            if tuple(qualityPOMDP.qualityPOMDPBelief.belief) not in buckets:
                buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] = 0
            buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] += 1
        E_b2 = 0
        for qualityPOMDP in qualityPOMDPs:
            if buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] > 0:
                E_b2 += qualityPOMDP.estimateBallotsToCompletionUsingFrontierFinding(6) * buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)]
                buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] = 0

        # print v_bar2,E_b2
        #Uncomment for partial sync
        # E_b2 = E_b1

        v_bar2,E_b2 = self.findBestStatePair(v_bar2,E_b2,c_2)

        self.currentState = self.inverseStateSpaceMapping[(E_b2, tau, v_bar2, c_2)]
        print "State after synchronization: " + str((E_b2,tau,v_bar2,c_2))

    '''
    Generates all possible admissible states in the state space.
    '''
    def createStateSpaceMapping(self):
        for tau in np.linspace(0, self.systemParameters.deadline, self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes + 1):
            for cost in np.linspace(1, self.systemParameters.numberOfPricePoints, self.systemParameters.numberOfPricePoints):
                for v_bar, validE_b in zip(self.possibleV_bar,self.validE_b[cost]):
                    for E_b in validE_b:
                        self.stateSpaceMapping.append((E_b, tau, v_bar, cost))
                        self.inverseStateSpaceMapping[(E_b, tau, v_bar, cost)] = self.numStates
                        self.numStates += 1

        self.stateSpaceMapping.append('TERMINAL')
        self.numStates += 1

        with open('stateSpaceMapping', 'w') as stateSpaceFile:
            for i in xrange(len(self.stateSpaceMapping)):
                stateSpaceFile.write(str(i) + '\t\t' + str(self.stateSpaceMapping[i]) + '\n')


    def createRewardAndTransitionFunction(self):
        fromStateArray = [[] for _ in xrange(self.numActions)]
        toStateArray = [[] for _ in xrange(self.numActions)]
        rewardArray = [[] for _ in xrange(self.numActions)]
        transProbArray = [[] for _ in xrange(self.numActions)]
        print "Total number of states: %d" % (self.numStates)
        for fromStateID,(E_b,tau,v_bar,cost) in enumerate(self.stateSpaceMapping[:-1]):
            fromStateArray[3].append(fromStateID)
            toStateArray[3].append(self.numStates - 1)
            rewardArray[3].append(self.systemParameters.timeReward * (tau == self.systemParameters.deadline) - self.systemParameters.numQuestions * self.systemParameters.value * (1 - v_bar) * 0.5)#0.1 * self.systemParameters.value * (E_b + (self.overallMaxBallotsDict[1] - self.overallMaxBallotsDict[cost]) - self.systemParameters.numQuestions * self.BVFunction[1][int(np.floor(v_bar*self.systemParameters.bvFunctionGranularity))])/self.maxBallots)
            transProbArray[3].append(1.0)
            if tau == self.systemParameters.deadline or v_bar == 1: #We can only submit at this point
                fromStateArray[0].append(fromStateID)
                toStateArray[0].append(self.numStates - 1)
                rewardArray[0].append(-10000000000)
                transProbArray[0].append(1.0)
                fromStateArray[1].append(fromStateID)
                toStateArray[1].append(self.numStates - 1)
                rewardArray[1].append(-10000000000)
                transProbArray[1].append(1.0)
                fromStateArray[2].append(fromStateID)
                toStateArray[2].append(self.numStates - 1)
                rewardArray[2].append(-10000000000)
                transProbArray[2].append(1.0)
                continue

            if E_b == 0:
                fromStateArray[0].append(fromStateID)
                toStateArray[0].append(self.numStates - 1)
                rewardArray[0].append(-10000000000)
                transProbArray[0].append(1.0)
                fromStateArray[1].append(fromStateID)
                toStateArray[1].append(self.numStates - 1)
                rewardArray[1].append(-10000000000)
                transProbArray[1].append(1.0)
                if cost == 1:
                    fromStateArray[2].append(fromStateID)
                    toStateArray[2].append(fromStateID)
                    rewardArray[2].append(-1000000000)
                    transProbArray[2].append(1.0)
                    continue
                t_S_Eb = self.validCrossCostTransitions[(E_b,v_bar,cost,cost-1)]
                toStateID = self.inverseStateSpaceMapping[(t_S_Eb, tau, v_bar, cost-1)]
                fromStateArray[2].append(fromStateID)
                toStateArray[2].append(toStateID)
                rewardArray[2].append(-10)
                transProbArray[2].append(1.0)
                continue

            for action in xrange(self.numActions):
                if action == 0:
                    t_S_Tau = tau + self.systemParameters.timeGranularityInMinutes
                    toStates = []
                    probs = []
                    rewards = []
                    for numBallots in np.linspace(0, E_b, E_b/self.systemParameters.ballotsToCompletionGranularity + 1):
                        (t_S_Eb, t_S_vBar) = self.validSameCostTransitions[(E_b,v_bar,cost,numBallots)]
                        toStateID = self.inverseStateSpaceMapping[(t_S_Eb, t_S_Tau, t_S_vBar, cost)]
                        if numBallots != E_b:
                            transProb = self.transitionProbabilityTable[cost][tau][0][numBallots]
                        else:
                            transProb = self.transitionProbabilityTable[cost][tau][1][numBallots]

                        if transProb > 0:
                            toStates.append(toStateID)
                            probs.append(float('%.4f' % transProb))
                            rewards.append(-1 * numBallots * cost)

                    if probs:
                        probs = list(normalize(np.array(probs)[np.newaxis,:], norm='l1')[0])

                    for t_S,transProb,reward in zip(toStates,probs,rewards):
                        fromStateArray[0].append(fromStateID)
                        toStateArray[0].append(t_S)
                        rewardArray[0].append(reward)
                        transProbArray[0].append(float(transProb))

                elif action == 1 or action == 2:
                    if (action == 1 and cost == self.systemParameters.numberOfPricePoints) or (action == 2 and cost == 1):
                        fromStateArray[action].append(fromStateID)
                        toStateArray[action].append(fromStateID)
                        rewardArray[action].append(-1000000000)
                        transProbArray[action].append(1.0)
                        continue

                    t_S_Eb = self.validCrossCostTransitions[(E_b,v_bar,cost,cost + (-1)**(action - 1))]
                    toStateID = self.inverseStateSpaceMapping[(t_S_Eb, tau, v_bar, cost  + (-1)**(action - 1))]
                    fromStateArray[action].append(fromStateID)
                    toStateArray[action].append(toStateID)
                    rewardArray[action].append(-10)#This lets us train the CostMDP as an undiscounted infinite horizon MDP (cyclical behavior in cost increase/decrease leads to infinite penalty)
                    transProbArray[action].append(1.0)


        for action in xrange(self.numActions):
            fromStateArray[action].append(self.numStates - 1)
            toStateArray[action].append(self.numStates - 1)
            rewardArray[action].append(0)
            transProbArray[action].append(1.0)

        self.T = (csr_matrix((transProbArray[0], (fromStateArray[0], toStateArray[0])), shape=(self.numStates, self.numStates)),
             csr_matrix((transProbArray[1], (fromStateArray[1], toStateArray[1])), shape=(self.numStates, self.numStates)),
             csr_matrix((transProbArray[2], (fromStateArray[2], toStateArray[2])), shape=(self.numStates, self.numStates)),
            csr_matrix((transProbArray[3], (fromStateArray[3], toStateArray[3])), shape=(self.numStates, self.numStates)))

        self.R = (csr_matrix((rewardArray[0], (fromStateArray[0], toStateArray[0])), shape=(self.numStates, self.numStates)),
             csr_matrix((rewardArray[1], (fromStateArray[1], toStateArray[1])), shape=(self.numStates, self.numStates)),
             csr_matrix((rewardArray[2], (fromStateArray[2], toStateArray[2])), shape=(self.numStates, self.numStates)),
            csr_matrix((rewardArray[3], (fromStateArray[3], toStateArray[3])), shape=(self.numStates, self.numStates)))


    def createValidStateTransitionTableUpdated(self,simulations=1):
        os.chdir(SystemParameters.path + 'TMDP')
        if os.path.exists('log/transitionTables/%s' % (self.systemParameters.stringifyForTransitionTable())):

            while True:
                try:
                    os.mkdir('locks/%s' % (self.systemParameters.stringifyForTransitionTable()))
                    break
                except OSError:
                    pass
            print "Reading valid state transition table."
            (validSameCostTransitions,validCrossCostTransitions) = pickle.load(open('log/transitionTables/%s' % (self.systemParameters.stringifyForTransitionTable())))
            os.rmdir('locks/%s' % (self.systemParameters.stringifyForTransitionTable()))
	    print "Read state transition table!"
            return (validSameCostTransitions,validCrossCostTransitions)

        print "Creating valid state transition table."
        validSameCostTransitions = {}
        validCrossCostTransitions = {} #If you increase or decrease cost, then the belief in the answer v_bar should remain exactly the same.

        os.chdir(SystemParameters.path + 'QualityPOMDP')
        qualityPOMDPPolicyList = {x:QualityPOMDPPolicy(self.systemParameters,value=self.systemParameters.value,price=x) for x in self.possibleCosts}
        tasks = []
        POMDPLookup = {}
        for _ in xrange(self.systemParameters.numQuestions):
            question = Question(question_answer=0)
            POMDP = QualityPOMDP(question,qualityPOMDPPolicyList[1])
            tasks.append(POMDP)
            POMDPLookup[question.question_id] = POMDP

        possibleCompletions = []
        for val in self.possibleV_bar[1:]:
            possibleCompletions.append(val - 1.0/self.systemParameters.completenessGranularity)

        bundlePOMDP = QualityPOMDP(None,qualityPOMDPPolicyList[1])
        controller = Controller([],type=self.systemParameters.controllerType,purpose=1,bundle=(possibleCompletions,self.systemParameters.difficultyDistribution.alpha,self.systemParameters.difficultyDistribution.beta,bundlePOMDP))

        for c_1 in self.possibleCosts:
            print 'Cost ' + str(c_1)
            for task in tasks:
                task.qualityPOMDPPolicy = qualityPOMDPPolicyList[c_1]
            for E_b1 in self.possibleE_b[c_1]:
                print '\tE_b ' + str(E_b1)
                for v_bar1 in self.possibleV_bar:
                    if v_bar1 == 1.0: #You can only transition to the TERMINAL state so there are no other valid transitions
                        continue
                    if E_b1 in self.validE_b[c_1][int(round(v_bar1 * self.systemParameters.completenessGranularity))]:
                        print '\t\tv_bar ' + str(v_bar1)
                        if (E_b1 + (self.overallMaxBallotsDict[1] - self.overallMaxBallotsDict[c_1]),v_bar1,1,0) in validSameCostTransitions:
                            numBallots = 0
                            while numBallots <= E_b1:
                                E_b2,v_bar2 = validSameCostTransitions[(E_b1 + (self.overallMaxBallotsDict[1] - self.overallMaxBallotsDict[c_1]),v_bar1,1,numBallots)]
                                if E_b2 - (self.overallMaxBallotsDict[1] - self.overallMaxBallotsDict[c_1]) >= 0:
                                    validSameCostTransitions[(E_b1,v_bar1,c_1,numBallots)] = (E_b2 - (self.overallMaxBallotsDict[1] - self.overallMaxBallotsDict[c_1]),v_bar2)
                                else:
                                    validSameCostTransitions[(E_b1,v_bar1,c_1,numBallots)] = (0,v_bar2)
                                numBallots += self.systemParameters.ballotsToCompletionGranularity
                        else:
                            A, B = 1, 1000000 #If v_bar1 is 0.0, then we use a highly skewed beta distribution.
                            if v_bar1 != 0.0:
                                A, B = self.reconstructDistribution(E_b1,v_bar1,c_1)#Reconstruct the beta distribution

                            reconstructedDistribution = beta(A,B)

                            #Discretize and bucket the questions into the beta distribution
                            cumulativeProbabilityAllocated = 0
                            taskCompletions = []
                            for val in self.possibleV_bar[1:]:
                                probInRange = (reconstructedDistribution.cdf(val) - cumulativeProbabilityAllocated)
                                tasksInRange = probInRange * self.systemParameters.numQuestions
                                tasksAllocated = int(np.floor(tasksInRange))
                                taskCompletions.extend([val - 1.0/self.systemParameters.completenessGranularity] * tasksAllocated)
                                tasksLeft = (tasksInRange - tasksAllocated)
                                probLeft = tasksLeft/float(self.systemParameters.numQuestions)
                                cumulativeProbabilityAllocated += (probInRange - probLeft)

                            taskCompletions.extend([self.possibleV_bar[-1] - 1.0/self.systemParameters.completenessGranularity] * (self.systemParameters.numQuestions - len(taskCompletions)))

                            #Do stuff for action = 0, same cost transitions (c_2 = c_1)
                            validSameCostTransitions[(E_b1,v_bar1,c_1,0)] = (E_b1,v_bar1)
                            if not E_b1 == 0:
                                v_bar2 = 0
                                for _ in xrange(simulations):
                                    for task,taskCompletion in zip(tasks,taskCompletions):
                                        task.qualityPOMDPBelief.setBelief((taskCompletion + 1)/2.0,
                                                        self.systemParameters.difficultyDistribution.alpha,
                                                        self.systemParameters.difficultyDistribution.beta)

                                    # controller = Controller(tasks,type=self.systemParameters.controllerType)
                                    controller.resetController(tasks)
                                    zeros = int(np.floor(self.systemParameters.ballotsToCompletionGranularity * calcAccuracy(self.systemParameters.averageGamma,self.systemParameters.difficultyDistribution.getMean()))) 
                                    ballots = np.random.permutation([0] * zeros + [1] * (self.systemParameters.ballotsToCompletionGranularity - zeros))
                                    for ballot in ballots:
                                        questionID = controller.assignQuestion()
                                        if not questionID:
                                            break
                                        questionID = questionID[0]
                                        POMDPLookup[questionID].qualityPOMDPBelief.updateBelief(ballot,self.systemParameters.averageGamma)
                                        controller.addAvailableQuestion(POMDPLookup[questionID],cached=0)
                                    v_bar2 += reduce(lambda x,y:x + y,map(lambda x: 2*x.qualityPOMDPBelief.v_max - 1,tasks)) / float(self.systemParameters.numQuestions)
                                v_bar2 = v_bar2/simulations
                                E_b2 = E_b1 - self.systemParameters.ballotsToCompletionGranularity
                                if not v_bar2 > v_bar1:
                                    v_bar2 = v_bar1
                                v_bar2, E_b2 = self.findBestStatePair(v_bar2,E_b2,c_1)
                                if E_b2 >= E_b1:
                                    v_bar2, E_b2 = self.adjustVBar(v_bar2,E_b1 - self.systemParameters.ballotsToCompletionGranularity,c_1)
                                validSameCostTransitions[(E_b1,v_bar1,c_1,self.systemParameters.ballotsToCompletionGranularity)] = (E_b2,v_bar2)
                                numBallots = 2*self.systemParameters.ballotsToCompletionGranularity
                                while numBallots <= E_b1:
                                    if not E_b2 == 0:
                                        E_b2, v_bar2 = validSameCostTransitions[(E_b2,v_bar2,c_1,self.systemParameters.ballotsToCompletionGranularity)]
                                    validSameCostTransitions[(E_b1,v_bar1,c_1,numBallots)] = (E_b2, v_bar2)
                                    numBallots += self.systemParameters.ballotsToCompletionGranularity

                        c_2 = c_1 + 1
                        if not c_2 > self.systemParameters.numberOfPricePoints: #Can't increase cost otherwise
                            v_bar2 = v_bar1
                            E_b2 = E_b1 - (self.overallMaxBallotsDict[c_1] - self.overallMaxBallotsDict[c_2])
                            if E_b2 <= 0.0:
                                E_b2 = 0
                            v_bar2, E_b2 = self.findBestStatePair(v_bar2, E_b2, c_2)

                            validCrossCostTransitions[(E_b1,v_bar1,c_1,c_2)] = E_b2
                            validCrossCostTransitions[(E_b2,v_bar1,c_2,c_1)] = E_b1

                        c_2 = c_1 - 1
                        if c_1 > 1: #Can't decrease cost otherwise
                            v_bar2 = v_bar1
                            E_b2 = E_b1 - (self.overallMaxBallotsDict[c_1] - self.overallMaxBallotsDict[c_2])
                            if E_b2 < 0.0:
                                E_b2 = 0
                            v_bar2, E_b2 = self.findBestStatePair(v_bar2, E_b2, c_2)

                            validCrossCostTransitions[(E_b1,v_bar1,c_1,c_2)] = E_b2
                            validCrossCostTransitions[(E_b2,v_bar1,c_2,c_1)] = E_b1


        print "Writing transition tables."
        os.chdir(SystemParameters.path + 'TMDP')
        pickle.dump((validSameCostTransitions,validCrossCostTransitions),open('log/transitionTables/%s' % (self.systemParameters.stringifyForTransitionTable()),'w'))
        return (validSameCostTransitions,validCrossCostTransitions)

