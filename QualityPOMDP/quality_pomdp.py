# from AgentHuntReleaseOriginal.ModelLearning.utils import *
# from AgentHuntReleaseOriginal.ModelLearning.genPOMDP import *
# from AgentHuntReleaseOriginal.Data import *
# from AgentHuntReleaseOriginal.Ballots import *

import sys
sys.path.insert(0, '../')

from worker_distribution import *
from numpy import var
from scipy.stats import beta

from os import getcwd
from itertools import product
from quality_pomdp_belief import *
from quality_pomdp_policy import *
from fixed_policy import *
import copy

import matplotlib.pyplot as plt
from scipy import stats

class QualityPOMDP(object):

    actions = range(0,3)

    def __init__(self,question,qualityPOMDPPolicy):
        self.question = question
        self.current_agent_action = -1

        self.systemParameters = qualityPOMDPPolicy.systemParameters

        self.policy = False
        if qualityPOMDPPolicy.policyType == 0:
            self.policy = True

        self.typeOfOtherPolicy = qualityPOMDPPolicy.policyType

        self.qualityPOMDPPolicy = qualityPOMDPPolicy

        self.qualityPOMDPBelief = QualityPOMDPBelief(alpha=self.systemParameters.difficultyDistribution.alpha,
                                                     beta=self.systemParameters.difficultyDistribution.beta)
        #print "Quality POMDP initialized."


    def findBestValue(self,hyperplanes):
        bestValue = -129837198273981231
        for hyperplane in hyperplanes:
            dontUse = False
            for (b, entry) in zip(self.qualityPOMDPBelief.belief, hyperplane):
                if b != 0 and entry == '*':
                    dontUse = True
                    break
            if dontUse:
                continue
            value = dot(self.qualityPOMDPBelief.belief, hyperplane)
            if value > bestValue:
                bestValue = value
        return bestValue

    def findBestAction(self):
        if not self.policy:
            if self.typeOfOtherPolicy == 1:
                if self.qualityPOMDPBelief.v_max < self.qualityPOMDPPolicy.threshold:
                    return 0
                return int(self.qualityPOMDPBelief.prediction + 1)
            else:
                if self.qualityPOMDPBelief.ballots_taken < self.qualityPOMDPPolicy.numBallots:
                    return 0
                return int(self.qualityPOMDPBelief.prediction + 1)
        bestValue = -1230981239102938019
        bestAction = 0 #Assume there is at least one action
        for action in QualityPOMDP.actions:
            value = self.findBestValue(self.qualityPOMDPPolicy.policy[action])
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

    '''
    Each time, simulate the POMDP until it submits, and note the number of ballots that it took to submit.
    Mode 1: Average Gamma, Current Estimate of Difficulty
    Mode 2: Average Gamma, Updated Estimates of Difficulty
    Mode 3: Sample Gamma, Current Estimate of Difficulty
    Mode 4: Sample Gamma, Updated Estimates of Difficulty

    Returns both the mean and the mode of the samples. Note that the mean tends to overestimate the value,
    while the mode tends to underestimate the value. The harder the questions, the better the mean performs,
    and the easier the questions, the better the mode performs.
    '''
    def estimateBallotsToCompletionBySimulatingPOMDP(self,simulations=100,mode=2):

        currentBelief = copy.deepcopy(self.qualityPOMDPBelief)
        estimatedBallotsToCompletion = []

        difficulty = self.qualityPOMDPBelief.getAverageDifficulty()
        gamma = self.systemParameters.averageGamma

        for simulation in xrange(simulations):
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
            estimatedBallotsToCompletion.append(0)
            #Take ballots until the POMDP submits
            while True:
                # Get action from POMDP, take a ballot, update belief, repeat
                bestAction = self.findBestAction()
                if bestAction > 0:
                    break
                estimatedBallotsToCompletion[-1] += 1
                if mode == 2:
                    difficulty = self.qualityPOMDPBelief.getAverageDifficulty()
                elif mode == 3:
                    gamma = self.systemParameters.workerDistribution.generateWorker()
                elif mode == 4:
                    gamma = self.systemParameters.workerDistribution.generateWorker()
                    difficulty = self.qualityPOMDPBelief.getAverageDifficulty()


                ballot = generateBallot(gamma,difficulty,self.qualityPOMDPBelief.prediction)
                self.qualityPOMDPBelief.updateBelief(ballot,gamma)

        self.qualityPOMDPBelief = currentBelief
        return (average(estimatedBallotsToCompletion),stats.mode(estimatedBallotsToCompletion)[0][0])

    '''
    Find terminating frontiers (where the POMDP submits) in the binary tree constructed by simulating future ballots.
    We specify the lookahead, and prune everything below it. The main idea behind pruning is that most of these events
    will be low probability (involving a great degree of confusion amongst workers), so the estimate will not be
    affected by much.

    This can be seen as taking the (pruned) expectation over the distribution of possible values for ballots to completion.
    '''
    def estimateBallotsToCompletionUsingFrontierFinding(self,lookahead=10,depth=0):
        if lookahead == 0:
            return depth
        accuracy = calcAccuracy(self.systemParameters.averageGamma,self.qualityPOMDPBelief.getAverageDifficulty())#self.qualityPOMDPBelief.calculateAccuracy(self.systemParameters.averageGamma)#,self.qualityPOMDPBelief.getAverageDifficulty())
        answer = self.qualityPOMDPBelief.prediction#(1 - self.qualityPOMDPBelief.v_max) * (1 - self.qualityPOMDPBelief.prediction) + self.qualityPOMDPBelief.prediction * self.qualityPOMDPBelief.v_max #self.qualityPOMDPBelief.prediction#
        currentBelief = copy.deepcopy(self.qualityPOMDPBelief)
        bestAction = self.findBestAction()
        if bestAction == 0:
            self.qualityPOMDPBelief.updateBelief(0,self.systemParameters.averageGamma)
            zero_side = self.estimateBallotsToCompletionUsingFrontierFinding(lookahead-1,depth+1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
            self.qualityPOMDPBelief.updateBelief(1,self.systemParameters.averageGamma)
            one_side = self.estimateBallotsToCompletionUsingFrontierFinding(lookahead-1,depth+1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
            return (1-answer) * (zero_side * accuracy + one_side * (1-accuracy)) + \
                   answer * (zero_side * (1-accuracy) + one_side * accuracy)
        else:
            return depth

    def estimateBallotsToCompletionUsingFrontierFindingOther(self,lookahead=10,depth=0):
        if lookahead == 0:
            return depth
        accuracy = calcAccuracy(self.systemParameters.averageGamma,self.qualityPOMDPBelief.getAverageDifficulty())#self.qualityPOMDPBelief.calculateAccuracy(self.systemParameters.averageGamma)#,self.qualityPOMDPBelief.getAverageDifficulty())
        answer = (1 - self.qualityPOMDPBelief.v_max) * (1 - self.qualityPOMDPBelief.prediction) + self.qualityPOMDPBelief.prediction * self.qualityPOMDPBelief.v_max #self.qualityPOMDPBelief.prediction#
        currentBelief = copy.deepcopy(self.qualityPOMDPBelief)
        bestAction = self.findBestAction()
        if bestAction == 0:
            self.qualityPOMDPBelief.updateBelief(0,self.systemParameters.averageGamma)
            zero_side = self.estimateBallotsToCompletionUsingFrontierFinding(lookahead-1,depth+1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
            self.qualityPOMDPBelief.updateBelief(1,self.systemParameters.averageGamma)
            one_side = self.estimateBallotsToCompletionUsingFrontierFinding(lookahead-1,depth+1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
            return (1-answer) * (zero_side * accuracy + one_side * (1-accuracy)) + \
                   answer * (zero_side * (1-accuracy) + one_side * accuracy)
        else:
            return depth


    '''
    Simulates the trajectory of the POMDP k steps into the future by generating ballots, and updating the POMDP using them.
    '''
    def simulatePOMDP(self,k):
        currentBelief = copy.deepcopy(self.qualityPOMDPBelief)
        for _ in xrange(k):
            difficulty = self.qualityPOMDPBelief.getAverageDifficulty()
            # Get action from POMDP, take a ballot, update belief, repeat
            bestAction = self.findBestAction()
            if bestAction > 0:
                break
            ballot = generateBallot(self.systemParameters.averageGamma,difficulty,self.qualityPOMDPBelief.prediction)
            self.qualityPOMDPBelief.updateBelief(ballot,self.systemParameters.averageGamma)

        v_max = self.qualityPOMDPBelief.v_max
        self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
        return v_max

    '''
    Simulates the trajectory of the POMDP into the future by generating ballots until it submits, and updating the POMDP using them.
    '''
    def simulatePOMDPUntilSubmit(self):
        currentBelief = copy.deepcopy(self.qualityPOMDPBelief)
        averageGamma = self.systemParameters.averageGamma
        numBallots = 0
        while True:
            difficulty = self.qualityPOMDPBelief.getAverageDifficulty()
            # Get action from POMDP, take a ballot, update belief, repeat
            bestAction = self.findBestAction()
            if bestAction > 0:
                break
            numBallots += 1
            ballot = generateBallot(averageGamma,difficulty,self.qualityPOMDPBelief.prediction)
            self.qualityPOMDPBelief.updateBelief(ballot,averageGamma)

        v_max = self.qualityPOMDPBelief.v_max
        self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
        return v_max,numBallots


    def expectedValue(self,lookahead):
        if lookahead == 0:
            return self.qualityPOMDPBelief.v_max
        accuracy = calcAccuracy(self.systemParameters.averageGamma,self.qualityPOMDPBelief.getAverageDifficulty())
        answer = self.qualityPOMDPBelief.prediction#(1 - self.qualityPOMDPBelief.v_max) * (1 - self.qualityPOMDPBelief.prediction) + self.qualityPOMDPBelief.prediction * self.qualityPOMDPBelief.v_max
        currentBelief = copy.deepcopy(self.qualityPOMDPBelief)
        bestAction = self.findBestAction()

        if bestAction == 0:
            self.qualityPOMDPBelief.updateBelief(0,self.systemParameters.averageGamma)
            zero_side = self.expectedValue(lookahead - 1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)


            self.qualityPOMDPBelief.updateBelief(1,self.systemParameters.averageGamma)
            one_side = self.expectedValue(lookahead - 1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)

            return (1-answer) * (zero_side * accuracy + one_side * (1-accuracy)) + \
                   answer * (zero_side * (1-accuracy) + one_side * accuracy)
        else:
            return self.qualityPOMDPBelief.v_max


    def expectedFractionalValue(self,currentValue,lookahead=3,depth=0):
        if lookahead == 0:
            return (self.qualityPOMDPBelief.v_max - currentValue)/float(depth)
        accuracy = calcAccuracy(self.systemParameters.averageGamma,self.qualityPOMDPBelief.getAverageDifficulty())
        answer = self.qualityPOMDPBelief.prediction#(1 - self.qualityPOMDPBelief.v_max) * (1 - self.qualityPOMDPBelief.prediction) + self.qualityPOMDPBelief.prediction * self.qualityPOMDPBelief.v_max#self.qualityPOMDPBelief.prediction
        currentBelief = copy.deepcopy(self.qualityPOMDPBelief)
        best_action = self.findBestAction()

        if best_action == 0:
            self.qualityPOMDPBelief.updateBelief(0,self.systemParameters.averageGamma)
            zero_side = self.expectedFractionalValue(currentValue,lookahead - 1,depth + 1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
            self.qualityPOMDPBelief.updateBelief(1,self.systemParameters.averageGamma)
            one_side = self.expectedFractionalValue(currentValue,lookahead - 1,depth + 1)
            self.qualityPOMDPBelief = copy.deepcopy(currentBelief)
            return (1-answer) * (zero_side * accuracy + one_side * (1-accuracy)) + \
                   answer * (zero_side * (1-accuracy) + one_side * accuracy)
        else:
            return (self.qualityPOMDPBelief.v_max - currentValue)/float(depth)