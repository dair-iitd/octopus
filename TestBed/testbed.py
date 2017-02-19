# import matplotlib
# matplotlib.use('TkAgg')

import sys
sys.path.insert(0, '../QualityPOMDP')
sys.path.insert(0, '../TMDP')
sys.path.insert(0, '../Controller')
sys.path.insert(0, '..')
from system_parameters import *
from difficulty_distribution import *
from worker_distribution import *
from question import *
from worker_skill_estimation import *
# from AgentHuntReleaseOriginal.ModelLearning.utils import *
# from AgentHuntReleaseOriginal.ModelLearning.genPOMDP import *
# from AgentHuntReleaseOriginal.Data import *
# from AgentHuntReleaseOriginal.Ballots import *

from cost_mdp import *
from fixed_ballot_cost_mdp import *
from quality_pomdp import *
from quality_pomdp_policy import *
from fixed_ballot_policy import *
from quality_pomdp_belief import *
from controller import *
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy, histogram, ks_2samp, ttest_ind, beta, norm
import cPickle as pickle
import copy
import time
import numpy as np
import glob
from multiprocessing import Pool, Process
#import seaborn as sns


class Testbed(object):


    def __init__(self,scale,plotting=False):

        self.realArrivalRatesPoisson = np.array([{x:scale*0.3854 for x in range(0,1441,1)},
                                     {x:scale*0.4597 for x in range(0,1441,1)},
                                     {x:scale*0.5375 for x in range(0,1441,1)},
                                     {x:scale*0.6349 for x in range(0,1441,1)},
                                     {x:scale*0.8448 for x in range(0,1441,1)},
                                     {x:scale*0.9913 for x in range(0,1441,1)}])

        self.bogusArrivalRatesPoisson = np.array([{x:0.2 for x in range(0,721,1)},
                                     {x:0.8 for x in range(0,721,1)},
                                     {x:1.4 for x in range(0,721,1)},
                                     {x:2.0 for x in range(0,721,1)},
                                     {x:2.6 for x in range(0,721,1)},
                                     {x:3.2 for x in range(0,721,1)}])

        self.systemParameters = SystemParameters(DifficultyDistribution(2.0,2.0),
                                            WorkerDistribution(2.0,0.5),
                                            numQuestions=500,
                                            value=200,
                                            timeGranularityInMinutes=30,
                                            ballotsToCompletionGranularity=10,
                                            completenessGranularity=100,
                                            numberOfPricePoints=6,
                                            numberOfSimulations=200,
                                            synchronizationGranularity=0,
                                            workerArrivalRates=self.realArrivalRatesPoisson)


        self.questions = []
        self.retentions = []
        self.workers = []
        self.arrivals = [{x:[] for x in xrange(1,self.systemParameters.numberOfPricePoints)} for _ in xrange(self.systemParameters.numberOfSimulations)]
        self.qualityPOMDPs = []
        if not plotting:
            print "Dumping meta data."
            self.metaDataDump()
            print "Initializing meta data."
            self.initializeMetaData()
            # self.ballots = [{(x,y.question_id):generateBallot(self.workers[x],y.question_difficulty,y.question_answer) for x in xrange(self.systemParameters.numWorkers) for y in self.questions} for _ in xrange(self.systemParameters.numberOfSimulations)]
	    print "Initializing questions."
            os.chdir(SystemParameters.path + 'QualityPOMDP')
            POMDPPolicy = QualityPOMDPPolicy(self.systemParameters,value=self.systemParameters.value,price=self.systemParameters.currentPrice)
            for question in self.questions:
                self.qualityPOMDPs.append(QualityPOMDP(question,POMDPPolicy))
            # self.simulateEverything()
        else:
            self.plotting()


    def initializeMetaData(self):
        os.chdir(SystemParameters.path + 'TestBed/data')
        with open('questions/%s' % (self.systemParameters.stringifyQuestionData())) as f:
            f.readline()
            for line in f:
                question_id,question_difficulty,question_answer = [float(x) for x in line.rstrip().split(",")]
                self.questions.append(Question(question_id=int(question_id),question_difficulty=question_difficulty,question_answer=int(question_answer)))

        with open('retentions/%d' % (self.systemParameters.numWorkers)) as f:
            f.readline()
            for line in f:
                _,worker_retention = [int(x) for x in line.rstrip().split(",")]
                self.retentions.append(worker_retention)

        with open('workers/%s' % (self.systemParameters.stringifyWorkerData())) as f:
            f.readline()
            for line in f:
                _,skill = [float(x) for x in line.rstrip().split(",")]
                self.workers.append(skill)
        print "\t Arrivals"
        numberOfArrivals = 100000
        numIntervals = 50
        for cost in xrange(1,self.systemParameters.numberOfPricePoints + 1):
            arrivals = []
            with open('arrivals/%s_%d' % (self.systemParameters.stringifyArrivalData(cost),numberOfArrivals)) as f:
                for line in f:
                    arrivals.append(int(line.rstrip()))

            arrivals = np.array(arrivals)[:self.systemParameters.numberOfSimulations*numIntervals].reshape((self.systemParameters.numberOfSimulations,numIntervals))
            for i,arrivalsForSingleSimulation in enumerate(arrivals):
                self.arrivals[i][cost] = list(arrivalsForSingleSimulation)


    def metaDataDump(self):
        #First create questions; only do this if we don't already have the questions created
        os.chdir(SystemParameters.path + 'TestBed/data')
        if not os.path.exists('questions/%s' %(self.systemParameters.stringifyQuestionData())):
            with open('questions/%s' % (self.systemParameters.stringifyQuestionData()),'w') as f:
                f.write("QuestionID,Difficulty,TrueAnswer\n")
                for i in xrange(self.systemParameters.numQuestions):
                    f.write("%d,%1.3f,%d\n" % (i,self.systemParameters.difficultyDistribution.generateDifficulty(),0))

        if not os.path.exists('retentions/%d' % (self.systemParameters.numWorkers)):
            with open('retentions/%d' % (self.systemParameters.numWorkers),'w') as f:
                f.write("WorkerID,BallotRetention\n")
                for i,retention in enumerate(np.random.geometric(p=0.2,size=self.systemParameters.numWorkers)):
                    f.write("%d,%d\n" % (i,retention))

        if not os.path.exists('workers/%s' % (self.systemParameters.stringifyWorkerData())):
            with open('workers/%s' % (self.systemParameters.stringifyWorkerData()),'w') as f:
                f.write("WorkerID,Skill\n")
                for i,skill in enumerate([self.systemParameters.workerDistribution.generateWorker() for _ in xrange(self.systemParameters.numWorkers)]):
                    f.write("%d,%3.4f\n" % (i,skill))

        numberOfArrivals = 100000
        for cost in xrange(1,self.systemParameters.numberOfPricePoints+1):
            if not os.path.exists('arrivals/%s_%d' % (self.systemParameters.stringifyArrivalData(cost),numberOfArrivals)):
                with open('arrivals/%s_%d' % (self.systemParameters.stringifyArrivalData(cost),numberOfArrivals),'w') as f:
                    for x in np.random.poisson(self.systemParameters.workerArrivalRates[cost - 1][0],numberOfArrivals):
                        f.write("%d\n" % (x))


    def simulateEverything(self):
        print "Setting up."
        deadlines = range(240,721,120)#range(30,330,30)
        # deadlines = [10,20,40,50,70,80,100,110,130,140,160,170,190,200]
        controllers = [1]
        synchronizations = [0]
        num_ballots_per_task = [1,3,5,7]

        for controllerType in controllers:
            self.systemParameters.controllerType = controllerType
            # self.simulateStaticCosts(deadlines)
            for deadline in deadlines:
                for num in num_ballots_per_task:
                    self.simulateGao(deadline,num)
                # for synchronizationGranularity in synchronizations:
                    # self.systemParameters.synchronizationGranularity = synchronizationGranularity
                    # self.simulateOurSystem(deadline)
                    # self.simulatedFixed(deadline)


    def simulateStaticCosts(self,deadlines):
        self.systemParameters.deadline = deadlines[-1]
        numIntervals = self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes

        totalMoneySpentStaticCost = {x:{y:[] for y in deadlines} for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        totalBallotsTakenStaticCost = {x:{y:[] for y in deadlines} for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        qualityAchievedStaticCost = {x:{y:[] for y in deadlines} for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        accuracyAchievedStaticCost = {x:{y:[] for y in deadlines} for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}


        for cost in range(1,self.systemParameters.numberOfPricePoints + 1):
            print "Static Cost: " + str(cost)
            os.chdir(SystemParameters.path + 'QualityPOMDP')
            self.systemParameters.currentPrice = cost
            POMDPPolicy = QualityPOMDPPolicy(self.systemParameters,value=self.systemParameters.value,price=self.systemParameters.currentPrice)
            for questionPOMDP in self.qualityPOMDPs:
                questionPOMDP.qualityPOMDPPolicy = POMDPPolicy
                questionPOMDP.qualityPOMDPBelief.resetBelief()

            controllerMain = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)

            for sim in xrange(self.systemParameters.numberOfSimulations):
                print "************************"
                print "Simulation: " + str(sim)
                print "************************"
                totalMoneySpentStaticCost[cost][self.systemParameters.deadline].append(0)
                totalBallotsTakenStaticCost[cost][self.systemParameters.deadline].append(0)
                for questionPOMDP in self.qualityPOMDPs:
                    questionPOMDP.qualityPOMDPBelief.resetBelief()
                controller = copy.deepcopy(controllerMain)
                workerSkillEstimator = WorkerSkillEstimator(self.systemParameters)
                workerRetentions = copy.deepcopy(self.retentions)
                workerArrivals = copy.deepcopy(self.arrivals[sim])
                workerBallots = set()#copy.deepcopy(self.ballots[sim])
                for tau in np.linspace(0, self.systemParameters.deadline, numIntervals + 1):
                    print "-----------------------"
                    print "Time: " + str(tau)
                    print "-----------------------"
                    if tau in deadlines:
                        print "%d minute deadline hit!" % tau
                        print "Money Spent: " + str(totalMoneySpentStaticCost[cost][self.systemParameters.deadline][-1]) + " units"
                        print "Ballots Taken: " + str(totalBallotsTakenStaticCost[cost][self.systemParameters.deadline][-1])
                        totalMoneySpentStaticCost[cost][tau].append(totalMoneySpentStaticCost[cost][self.systemParameters.deadline][-1])
                        totalBallotsTakenStaticCost[cost][tau].append(totalBallotsTakenStaticCost[cost][self.systemParameters.deadline][-1])
                        v_bar = (reduce(lambda x,y:x + y,map(lambda x: x.qualityPOMDPBelief.v_max,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions))
                        qualityAchievedStaticCost[cost][tau].append(v_bar)
                        print "Average Quality: " + "%.3f" % (v_bar)
                        accuracy = np.sum([1 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max != 0.5 and x.qualityPOMDPBelief.prediction == 0] + [0.5 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max == 0.5]) / float(self.systemParameters.numQuestions)
                        print "Average Accuracy: " + "%.3f" % (accuracy)
                        accuracyAchievedStaticCost[cost][tau].append(accuracy)
                        if tau == self.systemParameters.deadline:
                            break

                    print "Relearning gammas."
                    workerSkillEstimator.relearnGammas()

                    arrivals = workerArrivals[self.systemParameters.currentPrice].pop()
                    whoIsWorking = {}
                    #Allocate workers
                    workerUnderConsideration = -1
                    while arrivals > 0:
                        workerUnderConsideration += 1
                        if workerUnderConsideration == len(workerRetentions):
                            workerRetentions = copy.deepcopy(self.retentions)
                            workerUnderConsideration = 0
                        if workerRetentions[workerUnderConsideration] <= 0:
                            continue
                        if workerRetentions[workerUnderConsideration] <= arrivals:
                            arrivals = arrivals - workerRetentions[workerUnderConsideration]
                            whoIsWorking[workerUnderConsideration] = workerRetentions[workerUnderConsideration]
                            workerRetentions[workerUnderConsideration] = 0
                        else:
                            whoIsWorking[workerUnderConsideration] = arrivals
                            workerRetentions[workerUnderConsideration] -= arrivals
                            arrivals = 0

                    arrivalListWithWorkerID = []
                    for worker in whoIsWorking:
                        for _ in xrange(whoIsWorking[worker]):
                            arrivalListWithWorkerID.append(worker)

                    arrivalsUsed = 0

                    for idx in xrange(len(arrivalListWithWorkerID)):
                        worker = arrivalListWithWorkerID[idx]
                        questionToAssign = controller.assignQuestion()
                        if not questionToAssign:
                            continue
                        arrivalsUsed += 1
                        questionToAssign = questionToAssign[0]

                        if (worker,questionToAssign) in workerBallots:
                            for otherWorker in whoIsWorking.keys():
                                if (otherWorker,questionToAssign) not in workerBallots:
                                    worker = otherWorker
                                    break

                        ballot = generateBallotModified(self.workers[worker],self.questions[questionToAssign].question_difficulty,
                                                            self.questions[questionToAssign].question_answer,worker,questionToAssign,sim)
                        workerBallots.add((worker,questionToAssign))
                        workerSkillEstimator.addBallot(ballot,worker,questionToAssign)
                        self.qualityPOMDPs[questionToAssign].qualityPOMDPBelief.updateBelief(ballot,workerSkillEstimator.getWorkerGamma(worker))
                        controller.addAvailableQuestion(self.qualityPOMDPs[questionToAssign])
                        totalMoneySpentStaticCost[cost][self.systemParameters.deadline][-1] += self.systemParameters.currentPrice


                    print "Arrivals: " + str(arrivalsUsed)
                    totalBallotsTakenStaticCost[cost][self.systemParameters.deadline][-1] += arrivalsUsed

        os.chdir(SystemParameters.path + 'TestBed/results')
        for cost in xrange(1,self.systemParameters.numberOfPricePoints + 1):
            for deadline in deadlines:
                self.systemParameters.deadline = deadline
                averageMoneySpent = np.average(totalMoneySpentStaticCost[cost][deadline])
                averageBallotsTaken = np.average(totalBallotsTakenStaticCost[cost][deadline])
                averageQualityAchieved = np.average(qualityAchievedStaticCost[cost][deadline])
                averageAccuracyAchieved = np.average(accuracyAchievedStaticCost[cost][deadline])
                realUtilityStatic = [-1*(money + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracy)) for money,accuracy in zip(totalMoneySpentStaticCost[cost][deadline],accuracyAchievedStaticCost[cost][deadline])]

                run = 1
                while os.path.exists('static/%d|%s_%d' % (cost,self.systemParameters.stringify(),run)):
                    run += 1
                with open('static/%d|%s_%d.results' % (cost,self.systemParameters.stringify(),run),'w') as f:
                    f.write(self.systemParameters.prettyReturn())
                    f.write("----------------------------" + "\n")
                    f.write("Statistics: Static Strategy (Cost %d)" % (int(cost)) + "\n")
                    f.write("----------------------------" + "\n")
                    f.write("Average Money Spent: " + str(averageMoneySpent) + "\n")
                    f.write("Average Ballots Taken: " + str(averageBallotsTaken) + "\n")
                    f.write("Average Quality Achieved: " + str(averageQualityAchieved) + "\n")
                    f.write("Average Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageQualityAchieved))) + "\n")
                    f.write("Average Accuracy Achieved: " + str(averageAccuracyAchieved) + "\n")
                    f.write("Average Real Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageAccuracyAchieved))) + "\n")
                    mean, sigma = np.mean(accuracyAchievedStaticCost[cost][deadline]), np.std(accuracyAchievedStaticCost[cost][deadline],ddof=1)
                    f.write("95% Confidence Interval on Accuracy: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
                    f.write("90% Confidence Interval on Accuracy: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")
                    mean, sigma = np.mean(realUtilityStatic), np.std(realUtilityStatic,ddof=1)
                    f.write("95% Confidence Interval on Real Utility: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
                    f.write("90% Confidence Interval on Real Utility: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")


                with open('static/%d|%s_%d.pickle' % (cost,self.systemParameters.stringify(),run),'w') as f:
                    pickle.dump((totalMoneySpentStaticCost[cost][deadline],totalBallotsTakenStaticCost[cost][deadline],qualityAchievedStaticCost[cost][deadline],accuracyAchievedStaticCost[cost][deadline]),f)



    def simulateOurSystem(self,deadline,outputTracingQualityGraphs=False):
        self.systemParameters.deadline = deadline
        self.systemParameters.currentPrice = 1

        print "Creating Cost MDP"
        costMDP = CostMDP(self.systemParameters)

        totalMoneySpentDynamic = []
        totalBallotsTakenDynamic = []
        qualityAchievedDynamic = []
        accuracyAchievedDynamic = []

        numIntervals = self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes

        if outputTracingQualityGraphs:
            costMDPEstimatesOfE_bAndV_bar = [[] for _ in xrange(numIntervals)]
            trueValuesOfE_bAndV_bar = [[] for _ in xrange(numIntervals)]

        os.chdir(SystemParameters.path + 'QualityPOMDP')
        qualityPOMDPPolicyList = {x:QualityPOMDPPolicy(self.systemParameters,value=self.systemParameters.value,price=x) for x in range(1,self.systemParameters.numberOfPricePoints + 1)}
        for qualityPOMDP in self.qualityPOMDPs:
            qualityPOMDP.qualityPOMDPBelief.resetBelief()
            qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[self.systemParameters.currentPrice]

        controllerMain = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)
        synchronizationTimes = np.linspace(0,self.systemParameters.deadline,numIntervals + 1)[::self.systemParameters.synchronizationGranularity][1:] if self.systemParameters.synchronizationGranularity > 0 else []

        for sim in xrange(self.systemParameters.numberOfSimulations):
            print "************************"
            print "Simulation: " + str(sim)
            print "************************"
            totalMoneySpentDynamic.append(0)
            totalBallotsTakenDynamic.append(0)
            self.systemParameters.currentPrice = 1
            costMDP.resetState()
            os.chdir(SystemParameters.path + 'QualityPOMDP')
            for qualityPOMDP in self.qualityPOMDPs:
                qualityPOMDP.qualityPOMDPBelief.resetBelief()
                qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[self.systemParameters.currentPrice]

            controller = copy.deepcopy(controllerMain)

            workerSkillEstimator = WorkerSkillEstimator(self.systemParameters)
            workerRetentions = copy.deepcopy(self.retentions)
            workerArrivals = copy.deepcopy(self.arrivals[sim])
            workerBallots = set()#copy.deepcopy(self.ballots[sim])
            for tau in np.linspace(0, self.systemParameters.deadline, numIntervals + 1):
                print "-----------------------"
                print "Time: " + str(tau)
                print "-----------------------"

                if costMDP.currentState == costMDP.numStates - 1 or tau == self.systemParameters.deadline:
                    print "Reached terminal state." if costMDP.currentState == costMDP.numStates - 1 else "Deadline hit."
                    print "Money Spent: " + str(totalMoneySpentDynamic[-1]) + " units"
                    print "Ballots Taken: " + str(totalBallotsTakenDynamic[-1])
                    v_bar = (reduce(lambda x,y:x + y,map(lambda x: x.qualityPOMDPBelief.v_max,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions))
                    qualityAchievedDynamic.append(v_bar)
                    print "Quality: " + "%.3f" % (v_bar)
                    accuracy = np.sum([1 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max != 0.5 and x.qualityPOMDPBelief.prediction == 0] + [0.5 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max == 0.5]) / float(self.systemParameters.numQuestions)
                    print "Accuracy: " + "%.3f" % (accuracy)
                    accuracyAchievedDynamic.append(accuracy)
                    break
                os.chdir(SystemParameters.path + 'QualityPOMDP')
                E_b1, tau, v_bar1, c_1 = costMDP.stateSpaceMapping[costMDP.currentState]
                print "Current State: " + str((E_b1, tau, v_bar1, c_1))

                if (self.systemParameters.synchronizationGranularity == 0 and outputTracingQualityGraphs):
                    costMDPEstimatesOfE_bAndV_bar[int(tau/self.systemParameters.timeGranularityInMinutes)].append((E_b1,v_bar1))
                    v_bar2 = reduce(lambda x,y:x + y,map(lambda x: 2*x.qualityPOMDPBelief.v_max - 1,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions)
                    buckets = {}
                    for qualityPOMDP in self.qualityPOMDPs:
                        if tuple(qualityPOMDP.qualityPOMDPBelief.belief) not in buckets:
                            buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] = 0
                        buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] += 1
                    E_b2 = 0
                    for qualityPOMDP in self.qualityPOMDPs:
                        if buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] > 0:
                            E_b2 += qualityPOMDP.estimateBallotsToCompletionUsingFrontierFinding(int(np.ceil(costMDP.maxBallots))) * buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)]
                            buckets[tuple(qualityPOMDP.qualityPOMDPBelief.belief)] = 0
                    trueValuesOfE_bAndV_bar[int(tau/self.systemParameters.timeGranularityInMinutes)].append((E_b2,v_bar2))

                if tau in synchronizationTimes:
                    print "Synchronizing."
                    costMDP.synchronizeState(self.qualityPOMDPs)

                print "Relearning gammas."
                workerSkillEstimator.relearnGammas()

                updateControllerAndPolicy = False
                if costMDP.getBestAction() == 1 or costMDP.getBestAction() == 2:
                    updateControllerAndPolicy = True

                #Start by getting the best action in the cost MDP
                while costMDP.getBestAction() == 1 or costMDP.getBestAction() == 2:
                    print "Decreasing price." if costMDP.getBestAction() == 2 else "Increasing price."
                    self.systemParameters.currentPrice -= (-1)**costMDP.getBestAction()
                    costMDP.executeActionUpdated(costMDP.getBestAction())
                    E_b1, tau, v_bar1, c_1 = costMDP.stateSpaceMapping[costMDP.currentState]
                    print "Updated State: " + str((E_b1, tau, v_bar1, c_1))

                if updateControllerAndPolicy:
                    for questionPOMDP in self.qualityPOMDPs:
                        questionPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[self.systemParameters.currentPrice]
                    controller.resetController(self.qualityPOMDPs)

                bestAction = costMDP.getBestAction()
                if bestAction == 0: #We need to get ballots; first figure out the number of arrivals
                    arrivals = workerArrivals[self.systemParameters.currentPrice].pop()
                    whoIsWorking = {}
                    #Allocate workers
                    workerUnderConsideration = -1
                    while arrivals > 0:
                        workerUnderConsideration += 1
                        if workerUnderConsideration == len(workerRetentions):
                            workerRetentions = copy.deepcopy(self.retentions)
                            workerUnderConsideration = 0
                        if workerRetentions[workerUnderConsideration] <= 0:
                            continue
                        if workerRetentions[workerUnderConsideration] <= arrivals:
                            arrivals = arrivals - workerRetentions[workerUnderConsideration]
                            whoIsWorking[workerUnderConsideration] = workerRetentions[workerUnderConsideration]
                            workerRetentions[workerUnderConsideration] = 0
                        else:
                            whoIsWorking[workerUnderConsideration] = arrivals
                            workerRetentions[workerUnderConsideration] -= arrivals
                            arrivals = 0

                    arrivalListWithWorkerID = []
                    for worker in whoIsWorking:
                        for _ in xrange(whoIsWorking[worker]):
                            arrivalListWithWorkerID.append(worker)

                    arrivalsUsed = 0

                    if len(arrivalListWithWorkerID) > E_b1:
                        arrivalListWithWorkerID = arrivalListWithWorkerID[:int(E_b1)]
                    for idx in xrange(len(arrivalListWithWorkerID)):
                        worker = arrivalListWithWorkerID[idx]
                        questionToAssign = controller.assignQuestion()
                        if not questionToAssign:
                            temp = controller
                            for otherCost in xrange(self.systemParameters.currentPrice-1,0,-1):
                                for questionPOMDP in self.qualityPOMDPs:
                                    questionPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[otherCost]
                                controller = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)
                                questionToAssign = controller.assignQuestion()
                                if questionToAssign:
                                    break
                            controller = temp
                            for questionPOMDP in self.qualityPOMDPs:
                                questionPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[self.systemParameters.currentPrice]
                            if not questionToAssign:
                                continue
                        arrivalsUsed += 1
                        questionToAssign = questionToAssign[0]
                        if (worker,questionToAssign) in workerBallots:
                            for otherWorker in whoIsWorking.keys():
                                if (otherWorker,questionToAssign) not in workerBallots:
                                    worker = otherWorker
                                    break

                        ballot = generateBallotModified(self.workers[worker],self.questions[questionToAssign].question_difficulty,
                                                            self.questions[questionToAssign].question_answer,worker,questionToAssign,sim)
                        workerBallots.add((worker,questionToAssign))
                        workerSkillEstimator.addBallot(ballot,worker,questionToAssign)
                        self.qualityPOMDPs[questionToAssign].qualityPOMDPBelief.updateBelief(ballot,workerSkillEstimator.getWorkerGamma(worker))
                        controller.addAvailableQuestion(self.qualityPOMDPs[questionToAssign])
                        totalMoneySpentDynamic[-1] += self.systemParameters.currentPrice

                    costMDP.executeActionUpdated(bestAction,arrivalsUsed)
                    print "Taking action " + str(bestAction) + "."
                    print "Arrivals: " + str(arrivalsUsed)
                    totalBallotsTakenDynamic[-1] += arrivalsUsed
                else:
                    costMDP.executeActionUpdated(bestAction)


        if outputTracingQualityGraphs and self.systemParameters.synchronizationGranularity == 0:
            os.chdir(SystemParameters.path)
            pickle.dump((costMDPEstimatesOfE_bAndV_bar,trueValuesOfE_bAndV_bar),open('Plots/QualityOfTracking/%s.pickle' % (self.systemParameters.stringify()), 'w'))
            plt.plot(range(len(costMDPEstimatesOfE_bAndV_bar)),map(lambda x:np.mean([y[0] for y in x]),costMDPEstimatesOfE_bAndV_bar),label='Cost MDP Estimate')
            plt.plot(range(len(costMDPEstimatesOfE_bAndV_bar)),map(lambda x:np.mean([y[0] for y in x]),trueValuesOfE_bAndV_bar),label='True Estimate')
            plt.xlabel('Time step (interval size = %d minutes)' % (int(self.systemParameters.timeGranularityInMinutes)))
            plt.ylabel('Estimated ballots to completion')
            plt.title('%d tasks, %d minutes deadline, %d ballots granularity, %d price points' % (self.systemParameters.numQuestions,self.systemParameters.deadline,self.systemParameters.ballotsToCompletionGranularity,self.systemParameters.numberOfPricePoints))
            plt.legend(loc='best')
            plt.savefig('Plots/QualityOfTracking/EB_%s.eps' % (self.systemParameters.stringify()), format='eps', dpi=1000)
            plt.close()
            plt.plot(range(len(costMDPEstimatesOfE_bAndV_bar)),map(lambda x:np.mean([y[1] for y in x]),costMDPEstimatesOfE_bAndV_bar),label='Cost MDP Estimate')
            plt.plot(range(len(costMDPEstimatesOfE_bAndV_bar)),map(lambda x:np.mean([y[1] for y in x]),trueValuesOfE_bAndV_bar),label='True Value')
            plt.xlabel('Time step (interval size = %d minutes)' % (int(self.systemParameters.timeGranularityInMinutes)))
            plt.ylabel('Average completeness')
            plt.title('%d tasks, %d minutes deadline, %d ballots granularity, %d price points' % (self.systemParameters.numQuestions,self.systemParameters.deadline,self.systemParameters.ballotsToCompletionGranularity,self.systemParameters.numberOfPricePoints))
            plt.legend(loc='best')
            plt.savefig('Plots/QualityOfTracking/VBAR_%s.eps' % (self.systemParameters.stringify()), format='eps', dpi=1000)
            plt.close()


        averageMoneySpent = np.average(totalMoneySpentDynamic)
        averageBallotsTaken = np.average(totalBallotsTakenDynamic)
        averageQualityAchieved = np.average(qualityAchievedDynamic)
        averageAccuracyAchieved = np.average(accuracyAchievedDynamic)
        realUtilityDynamic = [-1*(money + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracy)) for money,accuracy in zip(totalMoneySpentDynamic,accuracyAchievedDynamic)]

        os.chdir(SystemParameters.path + 'TestBed/results')
        run = 1
        while os.path.exists('ours/%s_%d.results' % (self.systemParameters.stringify(),run)):
            run += 1
        with open('ours/%s_%d.results' % (self.systemParameters.stringify(),run),'w') as f:
            f.write(self.systemParameters.prettyReturn())
            f.write("----------------------------" + "\n")
            f.write("Statistics: Dynamic Strategy" + "\n")
            f.write("----------------------------" + "\n")
            f.write("Average Money Spent: " + str(averageMoneySpent) + "\n")
            f.write("Average Ballots Taken: " + str(averageBallotsTaken) + "\n")
            f.write("Average Quality Achieved: " + str(averageQualityAchieved) + "\n")
            f.write("Average Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageQualityAchieved))) + "\n")
            f.write("Average Accuracy Achieved: " + str(averageAccuracyAchieved) + "\n")
            f.write("Average Real Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageAccuracyAchieved))) + "\n")
            mean, sigma = np.mean(accuracyAchievedDynamic), np.std(accuracyAchievedDynamic,ddof=1)
            f.write("95% Confidence Interval on Accuracy: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
            f.write("90% Confidence Interval on Accuracy: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")
            mean, sigma = np.mean(realUtilityDynamic), np.std(realUtilityDynamic,ddof=1)
            f.write("95% Confidence Interval on Real Utility: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
            f.write("90% Confidence Interval on Real Utility: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")

        with open('ours/%s_%d.pickle' % (self.systemParameters.stringify(),run),'w') as f:
            pickle.dump((totalMoneySpentDynamic,totalBallotsTakenDynamic,qualityAchievedDynamic,accuracyAchievedDynamic),f)


    def simulateGao(self,deadline,ballotsPerTask):
        self.systemParameters.deadline = deadline
        self.systemParameters.currentPrice = 1

        os.chdir(SystemParameters.path + 'TestBed/data/gao_policies')
        price = pickle.load(open('%d_%d' % (ballotsPerTask,deadline)))

        totalMoneySpentGao = []
        totalBallotsTakenGao = []
        qualityAchievedGao = []
        accuracyAchievedGao = []

        numIntervals = self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes

        os.chdir(SystemParameters.path + 'QualityPOMDP')
        qualityPOMDPPolicy = FixedBallotPolicy(self.systemParameters,ballotsPerTask)
        for qualityPOMDP in self.qualityPOMDPs:
            qualityPOMDP.qualityPOMDPBelief.resetBelief()
            qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicy
            qualityPOMDP.policy = False
            qualityPOMDP.typeOfOtherPolicy = 2

        controllerMain = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)

        for sim in xrange(self.systemParameters.numberOfSimulations):
            print "************************"
            print "Simulation: " + str(sim)
            print "************************"
            remain_subtasks = self.systemParameters.numQuestions * ballotsPerTask
            totalMoneySpentGao.append(0)
            totalBallotsTakenGao.append(0)
            self.systemParameters.currentPrice = 1
            os.chdir(SystemParameters.path + 'QualityPOMDP')
            for qualityPOMDP in self.qualityPOMDPs:
                qualityPOMDP.qualityPOMDPBelief.resetBelief()

            controller = copy.deepcopy(controllerMain)

            workerSkillEstimator = WorkerSkillEstimator(self.systemParameters)
            workerRetentions = copy.deepcopy(self.retentions)
            workerArrivals = copy.deepcopy(self.arrivals[sim])
            workerBallots = set()#copy.deepcopy(self.ballots[sim])
            for tau in np.linspace(0, self.systemParameters.deadline, numIntervals + 1):
                print "-----------------------"
                print "Time: " + str(tau)
                print "-----------------------"

                if tau == self.systemParameters.deadline:
                    print "Deadline hit."
                    print "Money Spent: " + str(totalMoneySpentGao[-1]) + " units"
                    print "Ballots Taken: " + str(totalBallotsTakenGao[-1])
                    v_bar = (reduce(lambda x,y:x + y,map(lambda x: x.qualityPOMDPBelief.v_max,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions))
                    qualityAchievedGao.append(v_bar)
                    print "Quality: " + "%.3f" % (v_bar)
                    accuracy = np.sum([1 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max != 0.5 and x.qualityPOMDPBelief.prediction == 0] + [0.5 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max == 0.5]) / float(self.systemParameters.numQuestions)
                    print "Accuracy: " + "%.3f" % (accuracy)
                    accuracyAchievedGao.append(accuracy)
                    break
                os.chdir(SystemParameters.path + 'QualityPOMDP')

                print "Relearning gammas."
                workerSkillEstimator.relearnGammas()

                self.systemParameters.currentPrice = price[remain_subtasks][int(tau/self.systemParameters.timeGranularityInMinutes)]

                arrivals = workerArrivals[self.systemParameters.currentPrice].pop()
                whoIsWorking = {}
                #Allocate workers
                workerUnderConsideration = -1
                while arrivals > 0:
                    workerUnderConsideration += 1
                    if workerUnderConsideration == len(workerRetentions):
                        workerRetentions = copy.deepcopy(self.retentions)
                        workerUnderConsideration = 0
                    if workerRetentions[workerUnderConsideration] <= 0:
                        continue
                    if workerRetentions[workerUnderConsideration] <= arrivals:
                        arrivals = arrivals - workerRetentions[workerUnderConsideration]
                        whoIsWorking[workerUnderConsideration] = workerRetentions[workerUnderConsideration]
                        workerRetentions[workerUnderConsideration] = 0
                    else:
                        whoIsWorking[workerUnderConsideration] = arrivals
                        workerRetentions[workerUnderConsideration] -= arrivals
                        arrivals = 0

                arrivalListWithWorkerID = []
                for worker in whoIsWorking:
                    for _ in xrange(whoIsWorking[worker]):
                        arrivalListWithWorkerID.append(worker)

                arrivalsUsed = 0

                if len(arrivalListWithWorkerID) > remain_subtasks:
                    arrivalListWithWorkerID = arrivalListWithWorkerID[:remain_subtasks]
                for idx in xrange(len(arrivalListWithWorkerID)):
                    worker = arrivalListWithWorkerID[idx]
                    questionToAssign = controller.assignQuestion()
                    if not questionToAssign:
                        continue
                    arrivalsUsed += 1
                    questionToAssign = questionToAssign[0]
                    if (worker,questionToAssign) in workerBallots:
                        for otherWorker in whoIsWorking.keys():
                            if (otherWorker,questionToAssign) not in workerBallots:
                                worker = otherWorker
                                break

                    ballot = generateBallotModified(self.workers[worker],self.questions[questionToAssign].question_difficulty,
                                                        self.questions[questionToAssign].question_answer,worker,questionToAssign,sim)
                    workerBallots.add((worker,questionToAssign))
                    workerSkillEstimator.addBallot(ballot,worker,questionToAssign)
                    self.qualityPOMDPs[questionToAssign].qualityPOMDPBelief.updateBelief(ballot,workerSkillEstimator.getWorkerGamma(worker))
                    self.qualityPOMDPs[questionToAssign].qualityPOMDPBelief.ballots_taken += 1
                    controller.addAvailableQuestion(self.qualityPOMDPs[questionToAssign])
                    totalMoneySpentGao[-1] += self.systemParameters.currentPrice

                print "Arrivals: " + str(arrivalsUsed)
                totalBallotsTakenGao[-1] += arrivalsUsed


        averageMoneySpent = np.average(totalMoneySpentGao)
        averageBallotsTaken = np.average(totalBallotsTakenGao)
        averageQualityAchieved = np.average(qualityAchievedGao)
        averageAccuracyAchieved = np.average(accuracyAchievedGao)
        realUtilityGao = [-1*(money + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracy)) for money,accuracy in zip(totalMoneySpentGao,accuracyAchievedGao)]

        os.chdir(SystemParameters.path + 'TestBed/results')
        run = 1
        while os.path.exists('gao/%s_%d.results' % (self.systemParameters.stringify(),run)):
            run += 1
        with open('gao/%s_%d.results' % (self.systemParameters.stringify(),run),'w') as f:
            f.write(self.systemParameters.prettyReturn())
            f.write("----------------------------" + "\n")
            f.write("Statistics: Gao Strategy" + "\n")
            f.write("----------------------------" + "\n")
            f.write("Average Money Spent: " + str(averageMoneySpent) + "\n")
            f.write("Average Ballots Taken: " + str(averageBallotsTaken) + "\n")
            f.write("Average Quality Achieved: " + str(averageQualityAchieved) + "\n")
            f.write("Average Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageQualityAchieved))) + "\n")
            f.write("Average Accuracy Achieved: " + str(averageAccuracyAchieved) + "\n")
            f.write("Average Real Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageAccuracyAchieved))) + "\n")
            mean, sigma = np.mean(accuracyAchievedGao), np.std(accuracyAchievedGao,ddof=1)
            f.write("95% Confidence Interval on Accuracy: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
            f.write("90% Confidence Interval on Accuracy: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")
            mean, sigma = np.mean(realUtilityGao), np.std(realUtilityGao,ddof=1)
            f.write("95% Confidence Interval on Real Utility: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
            f.write("90% Confidence Interval on Real Utility: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")

        with open('gao/%s_%d.pickle' % (self.systemParameters.stringify(),run),'w') as f:
            pickle.dump((totalMoneySpentGao,totalBallotsTakenGao,qualityAchievedGao,accuracyAchievedGao),f)


    def simulateFixed(self,deadline):
        self.systemParameters.deadline = deadline
        self.systemParameters.currentPrice = 1
        print "Creating Fixed Ballots Cost MDP"
        costMDP = FixedBallotCostMDP(self.systemParameters)

        totalMoneySpentFixed = []
        totalBallotsTakenFixed = []
        qualityAchievedFixed = []
        accuracyAchievedFixed = []

        numIntervals = self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes

        os.chdir(SystemParameters.path + 'QualityPOMDP')
        qualityPOMDPPolicy = FixedPolicy(self.systemParameters,self.systemParameters.threshold)
        for qualityPOMDP in self.qualityPOMDPs:
            qualityPOMDP.qualityPOMDPBelief.resetBelief()
            qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicy
            qualityPOMDP.policy = False

        controllerMain = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)

        synchronizationTimes = np.linspace(0,self.systemParameters.deadline,numIntervals + 1)[::self.systemParameters.synchronizationGranularity][1:] if self.systemParameters.synchronizationGranularity > 0 else []
        for sim in xrange(self.systemParameters.numberOfSimulations):
            print "************************"
            print "Simulation: " + str(sim)
            print "************************"
            totalMoneySpentFixed.append(0)
            totalBallotsTakenFixed.append(0)
            self.systemParameters.currentPrice = 1

            costMDP.resetState()

            os.chdir(SystemParameters.path + 'QualityPOMDP')
            for qualityPOMDP in self.qualityPOMDPs:
                qualityPOMDP.qualityPOMDPBelief.resetBelief()
                qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicy

            controller = copy.deepcopy(controllerMain)#Controller(questionPOMDPs,type=self.systemParameters.controllerType)

            workerSkillEstimator = WorkerSkillEstimator(self.systemParameters)
            workerRetentions = copy.deepcopy(self.retentions)
            workerArrivals = copy.deepcopy(self.arrivals[sim])
            workerBallots = set()#copy.deepcopy(self.ballots[sim])
            for tau in np.linspace(0, self.systemParameters.deadline, numIntervals + 1):
                print "-----------------------"
                print "Time: " + str(tau)
                print "-----------------------"

                if costMDP.currentState == costMDP.numStates - 1 or tau == self.systemParameters.deadline:
                    print "Reached terminal state." if costMDP.currentState == costMDP.numStates - 1 else "Deadline hit."
                    print "Money Spent: " + str(totalMoneySpentFixed[-1]) + " units"
                    print "Ballots Taken: " + str(totalBallotsTakenFixed[-1])
                    v_bar = (reduce(lambda x,y:x + y,map(lambda x: x.qualityPOMDPBelief.v_max,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions))
                    qualityAchievedFixed.append(v_bar)
                    print "Quality: " + "%.3f" % (v_bar)
                    accuracy = np.sum([1 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max != 0.5 and x.qualityPOMDPBelief.prediction == 0]) / float(self.systemParameters.numQuestions)
                    print "Accuracy: " + "%.3f" % (accuracy)
                    accuracyAchievedFixed.append(accuracy)
                    break
                os.chdir(SystemParameters.path + 'QualityPOMDP')
                E_b1, tau, v_bar1, c_1 = costMDP.stateSpaceMapping[costMDP.currentState]
                print "Current State: " + str((E_b1, tau, v_bar1, c_1))

                if tau in synchronizationTimes:
                    print "Synchronizing."
                    costMDP.synchronizeState(self.qualityPOMDPs)

                print "Relearning gammas."
                workerSkillEstimator.relearnGammas()

                #Start by getting the best action in the cost MDP
                while costMDP.getBestAction() == 1 or costMDP.getBestAction() == 2:
                    print "Decreasing price." if costMDP.getBestAction() == 2 else "Increasing price."
                    self.systemParameters.currentPrice -= (-1)**costMDP.getBestAction()
                    costMDP.executeActionUpdated(costMDP.getBestAction())
                    E_b1, tau, v_bar1, c_1 = costMDP.stateSpaceMapping[costMDP.currentState]
                    print "Updated State: " + str((E_b1, tau, v_bar1, c_1))

                bestAction = costMDP.getBestAction()
                if bestAction == 0: #We need to get ballots; first figure out the number of arrivals
                    arrivals = workerArrivals[self.systemParameters.currentPrice].pop()
                    whoIsWorking = {}
                    #Allocate workers
                    workerUnderConsideration = -1
                    while arrivals > 0:
                        workerUnderConsideration += 1
                        if workerUnderConsideration == len(workerRetentions):
                            workerRetentions = copy.deepcopy(self.retentions)
                            workerUnderConsideration = 0
                        if workerRetentions[workerUnderConsideration] <= 0:
                            continue
                        if workerRetentions[workerUnderConsideration] <= arrivals:
                            arrivals = arrivals - workerRetentions[workerUnderConsideration]
                            whoIsWorking[workerUnderConsideration] = workerRetentions[workerUnderConsideration]
                            workerRetentions[workerUnderConsideration] = 0
                        else:
                            whoIsWorking[workerUnderConsideration] = arrivals
                            workerRetentions[workerUnderConsideration] -= arrivals
                            arrivals = 0

                    arrivalListWithWorkerID = []
                    for worker in whoIsWorking:
                        for _ in xrange(whoIsWorking[worker]):
                            arrivalListWithWorkerID.append(worker)

                    arrivalsUsed = 0

                    if len(arrivalListWithWorkerID) > E_b1:
                        arrivalListWithWorkerID = arrivalListWithWorkerID[:int(E_b1)]

                    for idx in xrange(len(arrivalListWithWorkerID)):
                        worker = arrivalListWithWorkerID[idx]
                        questionToAssign = controller.assignQuestion()
                        if not questionToAssign:
                            continue
                        arrivalsUsed += 1
                        questionToAssign = questionToAssign[0]
                        if (worker,questionToAssign) in workerBallots:
                            for otherWorker in whoIsWorking.keys():
                                if (otherWorker,questionToAssign) not in workerBallots:
                                    worker = otherWorker
                                    break

                        ballot = generateBallotModified(self.workers[worker],self.questions[questionToAssign].question_difficulty,
                                                            self.questions[questionToAssign].question_answer,worker,questionToAssign,sim)
                        workerBallots.add((worker,questionToAssign))
                        workerSkillEstimator.addBallot(ballot,worker,questionToAssign)
                        self.qualityPOMDPs[questionToAssign].qualityPOMDPBelief.updateBelief(ballot,workerSkillEstimator.getWorkerGamma(worker))
                        controller.addAvailableQuestion(self.qualityPOMDPs[questionToAssign])
                        totalMoneySpentFixed[-1] += self.systemParameters.currentPrice

                    costMDP.executeActionUpdated(bestAction,arrivalsUsed)
                    print "Taking action " + str(bestAction) + "."
                    print "Arrivals: " + str(arrivalsUsed)
                    totalBallotsTakenFixed[-1] += arrivalsUsed
                else:
                    costMDP.executeActionUpdated(bestAction)


        averageMoneySpent = np.average(totalMoneySpentFixed)
        averageBallotsTaken = np.average(totalBallotsTakenFixed)
        averageQualityAchieved = np.average(qualityAchievedFixed)
        averageAccuracyAchieved = np.average(accuracyAchievedFixed)
        realUtilityFixed = [-1*(money + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracy)) for money,accuracy in zip(totalMoneySpentFixed,accuracyAchievedFixed)]

        os.chdir(SystemParameters.path + 'TestBed/results')
        run = 1
        while os.path.exists('fixed/%s|%1.2f_%d.results' % (self.systemParameters.stringify(),self.systemParameters.threshold,run)):
            run += 1
        with open('fixed/%s|%1.2f_%d.results' % (self.systemParameters.stringify(),self.systemParameters.threshold,run),'w') as f:
            f.write(self.systemParameters.prettyReturn())
            f.write("----------------------------" + "\n")
            f.write("Statistics: Fixed Strategy" + "\n")
            f.write("----------------------------" + "\n")
            f.write("Average Money Spent: " + str(averageMoneySpent) + "\n")
            f.write("Average Ballots Taken: " + str(averageBallotsTaken) + "\n")
            f.write("Average Quality Achieved: " + str(averageQualityAchieved) + "\n")
            f.write("Average Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageQualityAchieved))) + "\n")
            f.write("Average Accuracy Achieved: " + str(averageAccuracyAchieved) + "\n")
            f.write("Average Real Utility Achieved: " + str(-1*(averageMoneySpent + self.systemParameters.numQuestions * self.systemParameters.value * (1 - averageAccuracyAchieved))) + "\n")
            mean, sigma = np.mean(accuracyAchievedFixed), np.std(accuracyAchievedFixed,ddof=1)
            f.write("95% Confidence Interval on Accuracy: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
            f.write("90% Confidence Interval on Accuracy: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")
            mean, sigma = np.mean(realUtilityFixed), np.std(realUtilityFixed,ddof=1)
            f.write("95% Confidence Interval on Real Utility: " + str(norm.interval(0.95,loc=mean,scale=sigma)) + "\n")
            f.write("90% Confidence Interval on Real Utility: " + str(norm.interval(0.90,loc=mean,scale=sigma)) + "\n")

        with open('fixed/%s|%1.2f_%d.pickle' % (self.systemParameters.stringify(),self.systemParameters.threshold,run),'w') as f:
            pickle.dump((totalMoneySpentFixed,totalBallotsTakenFixed,qualityAchievedFixed,accuracyAchievedFixed),f)


    def plotting(self):
        print "Plotting!"
        os.chdir(SystemParameters.path + 'TestBed/results/')
        deadlines = range(240,1441,60)#[120]#[30,60,90,120,150,180]#range(10,220,10) + [240,270,300]#xrange(30,330,30)
        controllers = [1]
        synchronizations = [0]
        run = 1
        #sns.set_style("whitegrid")

        # for controllerType in controllers:
        #     ourStrategy_1 = {x:[] for x in synchronizations}
        #     ourStrategy_2 = {x:[] for x in synchronizations}
        #     ourStrategy_bounds = {x:[] for x in synchronizations}
        #     self.systemParameters.controllerType = controllerType
        #     for deadline in deadlines:
        #         self.systemParameters.deadline = deadline
        #         for synchronizationGranularity in synchronizations:
        #             self.systemParameters.synchronizationGranularity = synchronizationGranularity
        #             with open('ours/%s_%d.results' % (self.systemParameters.stringify(),run)) as f:
        #                 for line in f:
        #                     if line.startswith("Average Util"):
        #                         ourStrategy_1[synchronizationGranularity].append(float(line.rstrip().split(": ")[1]))
        #                     elif line.startswith("Average Real"):
        #                         ourStrategy_2[synchronizationGranularity].append(float(line.rstrip().split(": ")[1]))
        #                     elif line.startswith("90% Confidence Interval on Real"):
        #                         lower, upper = line.rstrip().split(": ")[1].split(", ")
        #                         lower = float(lower.split("(")[-1])
        #                         upper = float(upper.split(")")[0])
        #                         ourStrategy_bounds[synchronizationGranularity].append((lower,upper))
        #             with open('ours/%s_%d.pickle' % (self.systemParameters.stringify(),run)) as f:
        #                 (totalMoneySpentDynamic,totalBallotsTakenDynamic,qualityAchievedDynamic,accuracyAchievedDynamic) = pickle.load(f)
        #                 #averageCost = np.array(totalMoneySpentDynamic)/np.array(totalBallotsTakenDynamic,dtype=np.float)
        #                 realUtilityDynamic = [-1*(money + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracy)) for money,accuracy in zip(totalMoneySpentDynamic,accuracyAchievedDynamic)]
        #                 #plt.scatter(x=np.arange(250),y=realUtilityDynamic)
        #                 #plt.show()
        #     for synchronizationGranularity in synchronizations:
        #         #plt.plot(deadlines,np.array(ourStrategy_1[synchronizationGranularity])/np.array(ourStrategy_1[synchronizationGranularity]),label='Dynamic (S), Internal Utility' if synchronizationGranularity == 1 else 'Dynamic (NS), Internal Utility',marker='o')
        #         #plt.plot(deadlines,np.array(ourStrategy_2[synchronizationGranularity])/np.array(ourStrategy_2[synchronizationGranularity]),label='Dynamic (S), Real Utility' if synchronizationGranularity == 1 else 'Dynamic (NS), Real Utility',marker='o')
        #         plt.plot(deadlines,np.array(ourStrategy_1[synchronizationGranularity]),label='Dynamic (S)' if synchronizationGranularity == 1 else 'Dynamic (NS)',marker='o')
        #         # plt.plot(deadlines,np.array(ourStrategy_2[synchronizationGranularity]),label='Dynamic (S)' if synchronizationGranularity == 1 else 'Dynamic (NS)',marker='o')
        #         # if controllerType == 1:
        #         #     plt.plot(deadlines,np.array(ourStrategy_1[synchronizationGranularity]),label='Dynamic (S), Greedy' if synchronizationGranularity == 1 else 'Dynamic (NS), Greedy',marker='o')
        #         # elif controllerType == 3:
        #         #     plt.plot(deadlines,np.array(ourStrategy_1[synchronizationGranularity]),label='Dynamic (S), Random' if synchronizationGranularity == 1 else 'Dynamic (NS), Random',marker='o')
        #         # elif controllerType == 4:
        #         #     plt.plot(deadlines,np.array(ourStrategy_1[synchronizationGranularity]),label='Dynamic (S), Random Robin' if synchronizationGranularity == 1 else 'Dynamic (NS), Random Robin',marker='o')
        #         #plt.errorbar(deadlines,np.array(ourStrategy_2[synchronizationGranularity])/np.array(ourStrategy_2[synchronizationGranularity]),yerr=(np.array(map(lambda x:x[0],ourStrategy_bounds[synchronizationGranularity])) - np.array(ourStrategy_2[synchronizationGranularity]))/np.array(ourStrategy_2[synchronizationGranularity]),label='Dynamic (S), Real Utility' if synchronizationGranularity == 1 else 'Dynamic (NS), Real Utility',marker='o')

        controllers = [1]
        staticStrategy_1 = {x:[] for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        staticStrategy_2 = {x:[] for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        staticStrategy_bounds = {x:[] for x in xrange(1,self.systemParameters.numberOfPricePoints + 1)}
        for controllerType in controllers:
            self.systemParameters.controllerType = controllerType
            for deadline in deadlines:
                self.systemParameters.deadline = deadline
                for cost in xrange(1,self.systemParameters.numberOfPricePoints + 1):
                    with open('static/%d|%s_%d.results' % (cost,self.systemParameters.stringify(),run)) as f:
                        for line in f:
                            if line.startswith("Average Utility"):
                                staticStrategy_1[cost].append(float(line.rstrip().split(": ")[1]))
                            elif line.startswith("Average Real"):
                                staticStrategy_2[cost].append(float(line.rstrip().split(": ")[1]))
                            elif line.startswith("90% Confidence Interval on Real"):
                                lower, upper = line.rstrip().split(": ")[1].split(", ")
                                lower = float(lower.split("(")[-1])
                                upper = float(upper.split(")")[0])
                                staticStrategy_bounds[cost].append((lower,upper))

            for cost in xrange(1,self.systemParameters.numberOfPricePoints + 1):
                plt.plot(deadlines,staticStrategy_1[cost],label='SC = %d' % (cost),marker='x')
                plt.plot(deadlines,staticStrategy_2[cost],label='SC = %d' % (cost),marker='x')

        #sns.plt.show()
        plt.title('Plot of Real Utility for 100 Questions (250 Repetitions)')
        plt.xlabel('Time Deadline (Minutes)')
        plt.ylabel('Real Utility')
        plt.legend(loc='lower right')
        plt.show()

    def __call__(self, arg, mode):
        if mode == 0:
            self.simulateStaticCosts(arg)
        elif mode == 1:
            self.simulateOurSystem(arg)


def learnQualityPOMDPPolicies():
    print "Learning policies."

    for value in [200]:
        l = [(value,x,y) for x in range(1,7) for y in [0.25,0.5,0.75,1.0,1.25,1.5]]
        p = Pool(4)
        p.map(wrapper,l)

def learnQualityPOMDPPolicy(value,price,averageGamma):
    print "Entering."
    systemParameters = SystemParameters(DifficultyDistribution(4.0,1.0),WorkerDistribution(2.0,0.5),workerArrivalRates= 4 * [1])
    systemParameters.averageGamma = averageGamma
    QualityPOMDPPolicy(systemParameters,value=value,price=price)
    return

def wrapper(args):
   return learnQualityPOMDPPolicy(*args)

def wrapperExperiment1(scale,deadline,sync,controller):
    testbed = Testbed(scale=scale,plotting=False)
    testbed.systemParameters.synchronizationGranularity = sync
    testbed.systemParameters.controllerType = controller
    testbed.systemParameters.deadline = deadline
    os.chdir(SystemParameters.path + 'TestBed/results')
    run = 1
    if not os.path.exists('ours/%s_%d.results' % (testbed.systemParameters.stringify(),run)):
    	testbed.simulateOurSystem(deadline)
    return

def wrapTheWrapperExperiment1(args):
    return wrapperExperiment1(*args)

def Experiment1():
    print "Setting up Experiment 1."

    deadlines = range(120,721,30)
    synchronizations = [0]
    controllers = [1,3,4]
    scales = [10]#range(6,15,2)

    parameters = [(w,x,y,z) for w in scales for x in deadlines for y in synchronizations for z in controllers]
    p = Pool(3)
    p.map(wrapTheWrapperExperiment1,parameters)

def wrapperExperiment1StaticPart1(scale,deadlines,controller):
    testbed = Testbed(scale=scale,plotting=False)
    testbed.systemParameters.controllerType = controller
    testbed.simulateStaticCosts(deadlines)
    return

def wrapTheWrapperExperiment1StaticPart1(args):
    return wrapperExperiment1StaticPart1(*args)

def Experiment1StaticPart1():
    print "Setting up Experiment 1 Static Part 1."
    deadlines = range(240,1441,60)
    controllers = [1,3,4]
    scales = range(6,15,2)

    parameters = [(w,deadlines,z) for w in scales for z in controllers]
    p = Pool(15) #1 experimental setting per thread
    p.map(wrapTheWrapperExperiment1StaticPart1,parameters)

def wrapperExperiment1StaticPart2(value,deadlines,controller):
    testbed = Testbed(scale=10,plotting=False)
    testbed.systemParameters.controllerType = controller
    testbed.systemParameters.value = value
    testbed.simulateStaticCosts(deadlines)
    return

def wrapTheWrapperExperiment1StaticPart2(args):
    return wrapperExperiment1StaticPart2(*args)

def Experiment1StaticPart2():
    print "Setting up Experiment 1 Static Part2."
    deadlines = range(240,1441,60)
    controllers = [1,3,4]
    values = range(400,1001,200)

    parameters = [(w,deadlines,z) for w in values for z in controllers]
    p = Pool(15) #1 experimental setting per thread
    p.map(wrapTheWrapperExperiment1StaticPart2,parameters)


def wrapperExperiment2FixedPart1(scale,deadline,sync,controller):
    testbed = Testbed(scale=scale,plotting=False)
    testbed.systemParameters.synchronizationGranularity = sync
    testbed.systemParameters.controllerType = controller
    testbed.systemParameters.deadline = deadline
    testbed.simulateFixed(deadline)			
    return

def wrapTheWrapperExperiment2FixedPart1(args):
    return wrapperExperiment2FixedPart1(*args)

def Experiment2FixedPart1():
    print "Setting up Experiment 2 Part 1."

    deadlines = range(240,1441,60)
    synchronizations = [0,1]
    controllers = [1,3,4]
    scales = range(6,15,2)

    parameters = [(w,x,y,z) for w in scales for x in deadlines for y in synchronizations for z in controllers]
    p = Pool(32)
    p.map(wrapTheWrapperExperiment2FixedPart1,parameters)


def wrapperExperiment2FixedPart2(value,deadline,sync,controller):
    testbed = Testbed(scale=10,plotting=False)
    testbed.systemParameters.synchronizationGranularity = sync
    testbed.systemParameters.controllerType = controller
    testbed.systemParameters.deadline = deadline
    testbed.simulateFixed(deadline)         
    return

def wrapTheWrapperExperiment2FixedPart2(args):
    return wrapperExperiment2FixedPart2(*args)

def Experiment2FixedPart2():
    print "Setting up Experiment 2 Part 2."

    deadlines = range(240,1441,60)
    synchronizations = [0,1]
    controllers = [1,3,4]
    values = range(400,1001,200)

    parameters = [(w,x,y,z) for w in values for x in deadlines for y in synchronizations for z in controllers]
    p = Pool(32)
    p.map(wrapTheWrapperExperiment2FixedPart2,parameters)


# Experiment1StaticPart1()
# Experiment1StaticPart2()
# Experiment2FixedPart1()
# Experiment2FixedPart2()
# Experiment1()
testbed = Testbed(scale=5)
testbed.simulateEverything()
# learnQualityPOMDPPolicies()
# wrapper((100,15,1.0))
