# import matplotlib
# matplotlib.use('TkAgg')

import sys
sys.path.insert(0, '../QualityPOMDP')
sys.path.insert(0, '../TMDP')
sys.path.insert(0, '../Controller')
sys.path.insert(0, '../DataCollection')
sys.path.insert(0, '..')
from system_parameters import *
from difficulty_distribution import *
from worker_distribution import *
from question import *
from functions import *
from worker_skill_estimation import *
# from AgentHuntReleaseOriginal.ModelLearning.utils import *
# from AgentHuntReleaseOriginal.ModelLearning.genPOMDP import *
# from AgentHuntReleaseOriginal.Data import *
# from AgentHuntReleaseOriginal.Ballots import *

from cost_mdp import *
from fixed_ballot_cost_mdp import *
from quality_pomdp import *
from quality_pomdp_policy import *
from quality_pomdp_belief import *
from fixed_ballot_policy import *
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
import time
import dateutil.parser
#import seaborn as sns
import datetime as dt
import boto
from boto.mturk.question import SelectionAnswer, Overview, QuestionForm, Question as MTurkQuestion, QuestionContent, FormattedContent, AnswerSpecification, FreeTextAnswer
from boto.mturk.connection import MTurkConnection, MTurkRequestError
from boto.mturk.qualification import Requirement,Qualifications,PercentAssignmentsApprovedRequirement,NumberHitsApprovedRequirement


class LiveOnlineExperiment(object):


    def __init__(self,scale=10,trueArrivals=True):

        self.trueArrivals = trueArrivals
        self.scale = scale
        self.realArrivalRatesPoisson = np.array([{x:scale*0.3854 for x in range(0,1441,1)},
                                     {x:scale*0.4597 for x in range(0,1441,1)},
                                     {x:scale*0.5375 for x in range(0,1441,1)},
                                     {x:scale*0.6349 for x in range(0,1441,1)},
                                     {x:scale*0.8448 for x in range(0,1441,1)},
                                     {x:scale*0.9913 for x in range(0,1441,1)}])

        self.systemParameters = SystemParameters(DifficultyDistribution(1.0,1.0),
                                            WorkerDistribution(2.0,0.5),
                                            numQuestions=500,
                                            value=200,
                                            timeGranularityInMinutes=15,
                                            ballotsToCompletionGranularity=10,
                                            completenessGranularity=100,
                                            numberOfPricePoints=6,
                                            numberOfSimulations=1,
                                            synchronizationGranularity=1,
                                            workerArrivalRates=self.realArrivalRatesPoisson)


        self.questions = []
        self.qualityPOMDPs = []
        self.groundTruth = {}
        self.tweets = {}

        print "Initializing meta data."
        self.initializeData()
        print "Initializing questions."
        os.chdir(SystemParameters.path + 'QualityPOMDP')
        POMDPPolicy = QualityPOMDPPolicy(self.systemParameters,value=self.systemParameters.value,price=self.systemParameters.currentPrice)
        for question in self.questions:
            self.qualityPOMDPs.append(QualityPOMDP(question,POMDPPolicy))

        self.IS_SANDBOX = False
        self.DEBUG = 1 #2 to print out all requests

        #Connect to MTurk
        self.mturk = connectToMTurk(self.IS_SANDBOX,self.DEBUG)

        printAccountBalance(self.mturk)

        self.pendingBallots = []
        self.workersToTasks = {}
        self.tasksToWorkers = {}
        self.HITToTasks = {}
        self.HITToPrices = {}
        self.workersToQualifications = {}
        self.taskIDToQualificationID = {}

        self.initializeQualifications()

    def initializeQualifications(self):
        os.chdir(SystemParameters.path + 'TestBed/data')

        name = 'qualifications.sandbox' if self.IS_SANDBOX else 'qualifications.live'

        if os.path.exists(name):
            with open(name) as f:
                for line in f:
                    task_id,qualification_id = line.rstrip().split(",")
                    self.taskIDToQualificationID[int(task_id)] = qualification_id
            return

        for task_id in range(self.systemParameters.numQuestions):
            qual = self.mturk.create_qualification_type(name="Check-norepeat-task-%d" % (task_id),description="Checking to see if you've already done this task!",status='Active')
            self.taskIDToQualificationID[task_id] = qual[0].QualificationTypeId

        with open(name,'w') as f:
            for key,val in self.taskIDToQualificationID.items():
                f.write("%d,%s\n" % (key,val))



    def initializeData(self):
        os.chdir(SystemParameters.path + 'DataCollection/combined')

        min_tweet = 2500
        max_tweet = 3000
        with open('../square_twitter_data.csv') as f:
            for line in f:
                split_list = line.rstrip().split(",")
                tweet_id,label,tweet = int(split_list[0]),int(split_list[1]),",".join(split_list[2:])
                if tweet_id >= min_tweet and tweet_id < max_tweet:
                    label = 1 - label
                    self.groundTruth[tweet_id - min_tweet] = label
                    self.tweets[tweet_id - min_tweet] = tweet
                    self.questions.append(Question(question_id=tweet_id-min_tweet,question_answer=label))
                if tweet_id > max_tweet + 1:
                    break

    def checkHITs(self): #Check if we need to post another HIT; only do this if there is no HIT available online anymore
        hits_available = 0
        os.chdir(SystemParameters.path + 'TestBed')
        print "Checking HITs"
        postAnotherHIT = True
        otherfile = open('FinishedHITs','w')
        hitsStillActive = []
        with open('ActiveHITs') as f:
            for line in f:
                hit = line.rstrip()
                print hit
                try:
                    hit_response = self.mturk.get_hit(hit,['HITDetail', 'HITAssignmentSummary'])
                    if int(hit_response[0].NumberOfAssignmentsAvailable) > 0:
                        hits_available += 1
                except MTurkRequestError:
                    pass
                try:
                    assignments = self.mturk.get_assignments(hit,page_size=100)
                    if assignments:
                        otherfile.write('%s\n' % (hit))
                    else:
                        hitsStillActive.append(hit)
                except MTurkRequestError:
                    hitsStillActive.append(hit)
        otherfile.close()
        with open('ActiveHITs','w') as f:
            f.write("\n".join(hitsStillActive))
            if hitsStillActive:
                f.write("\n")
        if hits_available >= 3:
            postAnotherHIT = False
        return postAnotherHIT


    def downloadHITs(self):
        os.chdir(SystemParameters.path + 'TestBed')
        print "Downloading HITs"
        otherfile = open('DownloadedHITs','a')
        with open('FinishedHITs') as f:
            for line in f:
                hit = line.rstrip()
                print hit
                try:
                    assignments = self.mturk.get_assignments(hit,page_size=100)
                    for assignment in assignments:
                        for answer in assignment.answers[0]:
                            self.pendingBallots.append((assignment.HITId,assignment.WorkerId,int(answer.qid),int(answer.fields[0])))
                    otherfile.write('%s\n' % (hit))
                except MTurkRequestError:
                    with open('ActiveHITs','a') as fp:
                        fp.write("%s\n" % (hit))
        otherfile.close()


    def postHITOnline(self, taskIDs, nextDecisionEpoch):

        titleOfHIT = "Sentiment Analysis for Tweets"
        descriptionOfHIT = "Choose the sentiment (positive/negative) that most closely matches that of the tweet shown. The HIT contains 10 tweets. Every time"\
                            " you complete a HIT a new one will be posted. This is part of a pricing study so future HITs may pay more or less for the same work. Fast approvals (within 24 hours)."
        keywordsForHIT = ["Sentiment", "Sentiment Analysis", "Twitter", "Categorization", "Classification",
                          "Binary Choice", "Multiple Choice", "Twitter Sentiment Analysis"]

        approvalDelay = 86400 #24 Hours

        durationPerHIT = 300 #5 Minutes

        lifetimeOfHIT = (nextDecisionEpoch - dt.datetime.now()).seconds# self.systemParameters.timeGranularityInMinutes #24 Hours
        if lifetimeOfHIT <= 30:
            return

        #In Dollars
        payment = self.systemParameters.currentPrice/100.0

        #Number of Max Assignments
        MaxAssignments = 1

        #Creating separate qualifications
        qualificationIDs = [self.taskIDToQualificationID[task_id] for task_id in taskIDs]
        qualifications = createQualifications(self.IS_SANDBOX,qualificationIDs)

        NumberOfHITsToPost = 1

        HIT_TypeID = createHITType(self.mturk,titleOfHIT,descriptionOfHIT,payment,durationPerHIT,keywordsForHIT,approvalDelay,qualifications)

        #Get all tweets
        listOfTweets = [self.tweets[task_id] for task_id in taskIDs]

        overviewTitle = '<b><font color="black" face="Helvetica" size="3"> Instructions </font></b>'
        overviewDescription = '<font color="black" face="Helvetica" size = "2">1. <i>Choose the sentiment (either positive or negative) that <b>best (most closely)</b> matches the ' \
                           "sentiment expressed in the given tweet.</i></font>" , \
                              '<font color="black" face="Helvetica" size = "2">2.<i> There are 10 separate sentences given to you below. The answers for separate sentences are not related. </i></font>', \
                              '<font color="black" face="Helvetica" size = "2">3.<i> Example: "That was the best day ever!" is a positive sentiment, while "I do not like that guy :/" is a negative sentiment. Please use your judgment! </i></font>'


        HIT_IDs = []
        for j in xrange(NumberOfHITsToPost):
            numberOfQuestions = 10
            questionForm = createQuestionForm(overviewTitle,overviewDescription,numberOfQuestions,listOfTweets,list(taskIDs))
            try:
                HIT = self.mturk.create_hit(hit_type=HIT_TypeID,lifetime=lifetimeOfHIT,questions=questionForm,max_assignments=MaxAssignments)
                HIT_IDs.append(HIT[0].HITId)
                print "Posted HIT %d" % (j+1)
            except MTurkRequestError:
                pass

        os.chdir(SystemParameters.path + 'TestBed')
        #Store the HIT_IDs
        with open('ActiveHITs','a') as f:
            for hit in HIT_IDs:
                print hit
                self.HITToPrices[hit] = self.systemParameters.currentPrice
                self.HITToTasks[hit] = taskIDs
                f.write(hit)
                f.write("\n")


    def runOurSystem(self):
        print "Setting up."
        deadline = 240

        self.systemParameters.controllerType = 1
        self.systemParameters.synchronizationGranularity = 1

        self.systemParameters.deadline = deadline
        self.systemParameters.currentPrice = 1

        print "Creating Cost MDP"
        costMDP = CostMDP(self.systemParameters)

        totalMoneySpentDynamic = 0
        totalBallotsTakenDynamic = 0
        accuracyAchievedDynamic = 0

        numIntervals = self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes

        os.chdir(SystemParameters.path + 'QualityPOMDP')
        qualityPOMDPPolicyList = {x:QualityPOMDPPolicy(self.systemParameters,value=self.systemParameters.value,price=x) for x in range(1,self.systemParameters.numberOfPricePoints + 1)}
        for qualityPOMDP in self.qualityPOMDPs:
            qualityPOMDP.qualityPOMDPBelief.resetBelief()
            qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[self.systemParameters.currentPrice]

        controller = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)
        synchronizationTimes = np.linspace(0,self.systemParameters.deadline,numIntervals + 1)[::self.systemParameters.synchronizationGranularity][1:] if self.systemParameters.synchronizationGranularity > 0 else []

        workerSkillEstimator = WorkerSkillEstimator(self.systemParameters)

        decisionEpochs = []
        for interval in range(numIntervals + 1):
            decisionEpochs.append(dt.datetime.now() + dt.timedelta(0,interval*self.systemParameters.timeGranularityInMinutes*60))

        scheduledTasks = []

        for tau in np.linspace(0, self.systemParameters.deadline, numIntervals + 1):
            print "-----------------------"
            print "Time: " + str(tau)
            print "-----------------------"

            if costMDP.currentState == costMDP.numStates - 1 or tau == self.systemParameters.deadline:
                print "Reached terminal state." if costMDP.currentState == costMDP.numStates - 1 else "Deadline hit."
                print "Money Spent: " + str(totalMoneySpentDynamic) + " units"
                print "Ballots Taken: " + str(totalBallotsTakenDynamic)
                qualityAchievedDynamic = (reduce(lambda x,y:x + y,map(lambda x: x.qualityPOMDPBelief.v_max,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions))
                print "Quality: " + "%.3f" % (qualityAchievedDynamic)
                accuracyAchievedDynamic = np.sum([1 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max != 0.5 and x.qualityPOMDPBelief.prediction == x.question.question_answer] + [0.5 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max == 0.5]) / float(self.systemParameters.numQuestions)
                print "Accuracy: " + "%.3f" % (accuracyAchievedDynamic)
                break

            nextDecisionEpoch = decisionEpochs[int(tau/self.systemParameters.timeGranularityInMinutes) + 1]

            os.chdir(SystemParameters.path + 'QualityPOMDP')
            E_b1, tau, v_bar1, c_1 = costMDP.stateSpaceMapping[costMDP.currentState]
            print "Current State: " + str((E_b1, tau, v_bar1, c_1))

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
                controller.recomputeController(self.qualityPOMDPs)

            bestAction = costMDP.getBestAction()
            if bestAction == 0: #We need to get ballots; first figure out the number of arrivals

                arrivalsUsed = 0
                os.chdir(SystemParameters.path + 'TestBed')
                with open('ActiveHITs') as f:
                    for line in f:
                        tasks = self.HITToTasks[line.rstrip()]
                        for task in tasks:
                            controller.addAvailableQuestion(self.qualityPOMDPs[task])

                with open('ActiveHITs','w'):
                    pass

                lastCheck = decisionEpochs[0]
                while ( (nextDecisionEpoch - dt.datetime.now()).seconds > 20):
                    if (dt.datetime.now() - lastCheck).seconds > 15:
                        print (nextDecisionEpoch - dt.datetime.now()).seconds
                        if not len(scheduledTasks) == 10: #Need to get some tasks from the controller in
                            for _ in range(10 - len(scheduledTasks)):
                                questionToAssign = controller.assignQuestion()
                                scheduledTasks.extend(questionToAssign)
                            if not len(scheduledTasks) == 10:
                                for otherCost in xrange(self.systemParameters.currentPrice-1,0,-1):
                                    for questionPOMDP in self.qualityPOMDPs:
                                        questionPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[otherCost]
                                    controller.recomputeController(self.qualityPOMDPs)
                                    for _ in range(10 - len(scheduledTasks)):
                                        questionToAssign = controller.assignQuestion()
                                        if not questionToAssign:
                                            break
                                        scheduledTasks.extend(questionToAssign)
                                    if len(scheduledTasks) == 10:
                                        break
                                for questionPOMDP in self.qualityPOMDPs:
                                    questionPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicyList[self.systemParameters.currentPrice]
                                controller.recomputeController(self.qualityPOMDPs)

                        #First check if there needs to be a HIT posted; this needs to be done in a loop every 15-20 seconds
                        post = self.checkHITs()
                        self.downloadHITs()
                        for hit,worker,task,ballot in self.pendingBallots:
                            arrivalsUsed += 1
                            if worker not in self.workersToTasks:
                                self.workersToTasks[worker] = set()
                            self.workersToTasks[worker].add(task)
                            if task not in self.tasksToWorkers:
                                self.tasksToWorkers[task] = set()
                            self.tasksToWorkers[task].add(worker)
                            try:
                                self.mturk.assign_qualification(self.taskIDToQualificationID[task], worker, value=1, send_notification=False)
                            except MTurkRequestError:
                                pass
                            workerSkillEstimator.addBallot(ballot,worker,task)
                            self.qualityPOMDPs[task].qualityPOMDPBelief.updateBelief(ballot,workerSkillEstimator.getWorkerGamma(worker))
                            controller.addAvailableQuestion(self.qualityPOMDPs[task])
                            totalMoneySpentDynamic += self.HITToPrices[hit] * 0.1
                        self.pendingBallots = []
                        if post and len(scheduledTasks) == 10:
                            self.postHITOnline(scheduledTasks,nextDecisionEpoch)
                            scheduledTasks = []
                        lastCheck = dt.datetime.now()
                        time.sleep(15)


                costMDP.executeActionUpdated(bestAction,arrivalsUsed)
                print "Taking action " + str(bestAction) + "."
                print "Arrivals: " + str(arrivalsUsed)
                totalBallotsTakenDynamic += arrivalsUsed
            else:
                costMDP.executeActionUpdated(bestAction)


        realUtilityDynamic = -1*(totalMoneySpentDynamic + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracyAchievedDynamic))
        print "Real Utility:"
        print realUtilityDynamic

    def runDai(self):
        print "Setting up."
        deadline = 240
        price = 3

        self.systemParameters.controllerType = 1

        self.systemParameters.deadline = deadline
        self.systemParameters.currentPrice = price


        totalMoneySpentDai = 0
        totalBallotsTakenDai = 0
        accuracyAchievedDai = 0

        numIntervals = self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes

        os.chdir(SystemParameters.path + 'QualityPOMDP')
        qualityPOMDPPolicy = QualityPOMDPPolicy(self.systemParameters,value=self.systemParameters.value,price=self.systemParameters.currentPrice)
        for qualityPOMDP in self.qualityPOMDPs:
            qualityPOMDP.qualityPOMDPBelief.resetBelief()
            qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicy

        controller = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)

        workerSkillEstimator = WorkerSkillEstimator(self.systemParameters)

        decisionEpochs = []
        for interval in range(numIntervals + 1):
            decisionEpochs.append(dt.datetime.now() + dt.timedelta(0,interval*self.systemParameters.timeGranularityInMinutes*60))

        scheduledTasks = []

        for tau in np.linspace(0, self.systemParameters.deadline, numIntervals + 1):
            print "-----------------------"
            print "Time: " + str(tau)
            print "-----------------------"

            if tau == self.systemParameters.deadline:
                print "Deadline hit."
                print "Money Spent: " + str(totalMoneySpentDai) + " units"
                print "Ballots Taken: " + str(totalBallotsTakenDai)
                qualityAchievedDai = (reduce(lambda x,y:x + y,map(lambda x: x.qualityPOMDPBelief.v_max,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions))
                print "Quality: " + "%.3f" % (qualityAchievedDai)
                accuracyAchievedDai = np.sum([1 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max != 0.5 and x.qualityPOMDPBelief.prediction == x.question.question_answer] + [0.5 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max == 0.5]) / float(self.systemParameters.numQuestions)
                print "Accuracy: " + "%.3f" % (accuracyAchievedDai)
                break

            nextDecisionEpoch = decisionEpochs[int(tau/self.systemParameters.timeGranularityInMinutes) + 1]

            os.chdir(SystemParameters.path + 'QualityPOMDP')

            print "Relearning gammas."
            workerSkillEstimator.relearnGammas()

            arrivalsUsed = 0
            os.chdir(SystemParameters.path + 'TestBed')
            with open('ActiveHITs') as f:
                for line in f:
                    tasks = self.HITToTasks[line.rstrip()]
                    for task in tasks:
                        controller.addAvailableQuestion(self.qualityPOMDPs[task])

            with open('ActiveHITs','w'):
                pass

            lastCheck = decisionEpochs[0]
            while ( (nextDecisionEpoch - dt.datetime.now()).seconds > 20):
                if (dt.datetime.now() - lastCheck).seconds > 15:
                    if not len(scheduledTasks) == 10: #Need to get some tasks from the controller in
                        for _ in range(10 - len(scheduledTasks)):
                            questionToAssign = controller.assignQuestion()
                            scheduledTasks.extend(questionToAssign)

                    #First check if there needs to be a HIT posted; this needs to be done in a loop every 15-20 seconds
                    post = self.checkHITs()
                    self.downloadHITs()
                    for hit,worker,task,ballot in self.pendingBallots:
                        arrivalsUsed += 1
                        if worker not in self.workersToTasks:
                            self.workersToTasks[worker] = set()
                        self.workersToTasks[worker].add(task)
                        if task not in self.tasksToWorkers:
                            self.tasksToWorkers[task] = set()
                        self.tasksToWorkers[task].add(worker)
                        try:
                            self.mturk.assign_qualification(self.taskIDToQualificationID[task], worker, value=1, send_notification=False)
                        except MTurkRequestError:
                            pass
                        workerSkillEstimator.addBallot(ballot,worker,task)
                        self.qualityPOMDPs[task].qualityPOMDPBelief.updateBelief(ballot,workerSkillEstimator.getWorkerGamma(worker))
                        controller.addAvailableQuestion(self.qualityPOMDPs[task])
                        totalMoneySpentDai += self.HITToPrices[hit] * 0.1
                    self.pendingBallots = []
                    if post and len(scheduledTasks) == 10:
                        self.postHITOnline(scheduledTasks,nextDecisionEpoch)
                        scheduledTasks = []
                    lastCheck = dt.datetime.now()
                    time.sleep(15)

                print "Arrivals: " + str(arrivalsUsed)
                totalBallotsTakenDai += arrivalsUsed

        realUtilityDai = -1*(totalMoneySpentDai + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracyAchievedDai))
        print "Real Utility:"
        print realUtilityDai


    def runGao(self):
        deadline = 60
        self.systemParameters.deadline = deadline
        self.systemParameters.currentPrice = 1

        remain_subtasks = self.systemParameters.numQuestions

        os.chdir(SystemParameters.path + 'TestBed/data/gao_policies')
        (_,price) = pickle.load(open('%d' % (deadline)))
        ballotsPerTask = 1

        totalMoneySpentGao = 0
        totalBallotsTakenGao = 0
        accuracyAchievedGao = 0

        numIntervals = self.systemParameters.deadline/self.systemParameters.timeGranularityInMinutes

        os.chdir(SystemParameters.path + 'QualityPOMDP')
        qualityPOMDPPolicy = FixedBallotPolicy(self.systemParameters,ballotsPerTask)
        for qualityPOMDP in self.qualityPOMDPs:
            qualityPOMDP.qualityPOMDPBelief.resetBelief()
            qualityPOMDP.qualityPOMDPPolicy = qualityPOMDPPolicy
            qualityPOMDP.policy = False
            qualityPOMDP.typeOfOtherPolicy = 2

        controller = Controller(self.qualityPOMDPs,type=self.systemParameters.controllerType)

        workerSkillEstimator = WorkerSkillEstimator(self.systemParameters)

        decisionEpochs = []
        for interval in range(numIntervals + 1):
            decisionEpochs.append(dt.datetime.now() + dt.timedelta(0,interval*self.systemParameters.timeGranularityInMinutes*60))

        scheduledTasks = []

        for tau in np.linspace(0, self.systemParameters.deadline, numIntervals + 1):
            print "-----------------------"
            print "Time: " + str(tau)
            print "-----------------------"

            if tau == self.systemParameters.deadline:
                print "Deadline hit."
                print "Money Spent: " + str(totalMoneySpentGao) + " units"
                print "Ballots Taken: " + str(totalBallotsTakenGao)
                qualityAchievedGao = (reduce(lambda x,y:x + y,map(lambda x: x.qualityPOMDPBelief.v_max,self.qualityPOMDPs)) / float(self.systemParameters.numQuestions))
                print "Quality: " + "%.3f" % (qualityAchievedGao)
                accuracyAchievedGao = np.sum([1 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max != 0.5 and x.qualityPOMDPBelief.prediction == x.question.question_answer] + [0.5 for x in self.qualityPOMDPs if x.qualityPOMDPBelief.v_max == 0.5]) / float(self.systemParameters.numQuestions)
                print "Accuracy: " + "%.3f" % (accuracyAchievedGao)
                break

            nextDecisionEpoch = decisionEpochs[int(tau/self.systemParameters.timeGranularityInMinutes) + 1]

            print "Relearning gammas."
            workerSkillEstimator.relearnGammas()

            self.systemParameters.currentPrice = price[remain_subtasks][int(tau/self.systemParameters.timeGranularityInMinutes)]

            arrivalsUsed = 0
            os.chdir(SystemParameters.path + 'TestBed')
            with open('ActiveHITs') as f:
                for line in f:
                    tasks = self.HITToTasks[line.rstrip()]
                    for task in tasks:
                        controller.addAvailableQuestion(self.qualityPOMDPs[task])

            with open('ActiveHITs','w'):
                pass

            lastCheck = decisionEpochs[0]
            while ( (nextDecisionEpoch - dt.datetime.now()).seconds > 20):
                if (dt.datetime.now() - lastCheck).seconds > 15:
                    print (nextDecisionEpoch - dt.datetime.now()).seconds
                    if not len(scheduledTasks) == 10: #Need to get some tasks from the controller in
                        for _ in range(10 - len(scheduledTasks)):
                            questionToAssign = controller.assignQuestion()
                            scheduledTasks.extend(questionToAssign)

                    #First check if there needs to be a HIT posted; this needs to be done in a loop every 15-20 seconds
                    post = self.checkHITs()
                    self.downloadHITs()
                    for hit,worker,task,ballot in self.pendingBallots:
                        arrivalsUsed += 1
                        if worker not in self.workersToTasks:
                            self.workersToTasks[worker] = set()
                        self.workersToTasks[worker].add(task)
                        if task not in self.tasksToWorkers:
                            self.tasksToWorkers[task] = set()
                        self.tasksToWorkers[task].add(worker)
                        try:
                            self.mturk.assign_qualification(self.taskIDToQualificationID[task], worker, value=1, send_notification=False)
                        except MTurkRequestError:
                            pass
                        workerSkillEstimator.addBallot(ballot,worker,task)
                        self.qualityPOMDPs[task].qualityPOMDPBelief.updateBelief(ballot,workerSkillEstimator.getWorkerGamma(worker))
                        self.qualityPOMDPs[task].qualityPOMDPBelief.ballots_taken += 1
                        controller.addAvailableQuestion(self.qualityPOMDPs[task])
                        totalMoneySpentGao += self.HITToPrices[hit] * 0.1
                    self.pendingBallots = []
                    if post and len(scheduledTasks) == 10:
                        self.postHITOnline(scheduledTasks,nextDecisionEpoch)
                        scheduledTasks = []
                    lastCheck = dt.datetime.now()
                    time.sleep(15)

            remain_subtasks -= arrivalsUsed


        realUtilityGao = -1*(totalMoneySpentGao + self.systemParameters.numQuestions * self.systemParameters.value * (1 - accuracyAchievedGao))
        print "Real Utility:"
        print realUtilityGao



def buildAnswers(answerTuple):
    """
    answerTuple should contain the choices for a particular question in the form of a tuple/list
    """

    answerList = zip(answerTuple,range(len(answerTuple)))
    selectionAnswer = SelectionAnswer(min=1,
                              max=1,
                              style='radiobutton',
                              selections=answerList,
                              type='text',
                              other=False)
    return selectionAnswer

def createQuestionForm(overviewTitle, overviewDescription, numberOfTweets, listOfTweets, listOfTweetIDs):
    """
    Create an overview for an MTurk HIT
    """

    #The Question Form should contain 1 overview and 3 odd questions
    questionForm = QuestionForm()

    #Define the Overview
    overview = Overview()
    Title = FormattedContent(overviewTitle)
    overview.append(Title)
    overviewDescription1 = FormattedContent(overviewDescription[0])
    overviewDescription2 = FormattedContent(overviewDescription[1])
    overviewDescription3 = FormattedContent(overviewDescription[2])
    overview.append(overviewDescription1)
    overview.append(overviewDescription2)
    overview.append(overviewDescription3)
    #Append the Overview to the Question Form
    questionForm.append(overview)

    #Create the Questions, and Add them
    for i in xrange(numberOfTweets):
        overview = Overview()
        questionTitle = FormattedContent('<font face="Helvetica" size="2"><b> Tweet #' + str(i + 1) + '</b></font>')
        overview.append(questionTitle)
        questionBody = FormattedContent('<font face="Helvetica" size="2">' + listOfTweets[i] + '</font>')
        overview.append(questionBody)
        #answerTuple = tuple([('<a href="https://wikipedia.org/en/' + y.replace(" ","_") + '" target="_blank">' + y + "</a>") for y in list(listOfAnswers[i])])
        #links = FormattedContent('<b>Links</b> | ' + answerTuple[0] + ' | ' + answerTuple[1])
        #overview.append(links)
        questionForm.append(overview)
        question = createQuestion(listOfTweetIDs[i],i,listOfTweets[i],["Positive","Negative"])
        questionForm.append(question)

    return questionForm


def createQuestion(identifier, displayName, questionText, answers):
    """
    Create a question
    """
    questionContent = QuestionContent()

    #questionTitle = FormattedContent('<font face="Helvetica" size="2"><b> Sentence #' + str(displayName + 1) + '</b></font>')
    #questionContent.append(questionTitle)

    instruction = 'Which of the following sentiments <i>best</i> matches the sentiment expressed in the above sentence?'

    #questionBody = FormattedContent('<font face="Helvetica" size="2">' + questionText + '</font>')
    #questionContent.append(questionBody)

    questionInstruction = FormattedContent('<font face="Helvetica" size="2">' + instruction + '</font>')
    questionContent.append(questionInstruction)

    #answerTuple = tuple([('<a href="https://wikipedia.org/en/' + y.replace(" ","_") + '" target="_blank">' + y + "</a>") for y in list(answers)])
    #links = FormattedContent('<b>Links</b> | ' + answerTuple[0] + ' | ' + answerTuple[1])
    #questionContent.append(links)

    displayName = "Question #" + str(displayName + 1)
    question = MTurkQuestion(identifier=identifier,
                        content=questionContent,
                        display_name=displayName,
                        answer_spec=AnswerSpecification(buildAnswers(answers)),
                        is_required=True)
    return question

def connectToMTurk(sandbox, debug):
    sandbox_host = 'mechanicalturk.sandbox.amazonaws.com'
    real_host = 'mechanicalturk.amazonaws.com'

    if sandbox:
        mturk = MTurkConnection(
            host = sandbox_host,
            debug = debug # debug = 2 prints out all requests.
        )
        print "Connected to the Sandbox"
    else:
        mturk = MTurkConnection(
            host = real_host,
            debug = debug # debug = 2 prints out all requests.
        )
        print "Connected to the Real Host"

    return mturk

def printAccountBalance(mturk):
    print "Account Balance:"
    print mturk.get_account_balance()


def createQualifications(sandbox,qualificationIDs):
    qualifications = Qualifications()
    for qualification_id in qualificationIDs[:8]:
        qualifications.add(Requirement(qualification_id,'DoesNotExist',required_to_preview=True))

    qualifications.add(PercentAssignmentsApprovedRequirement('GreaterThanOrEqualTo',90,True))
    qualifications.add(NumberHitsApprovedRequirement('GreaterThanOrEqualTo',100,True))#100

    return qualifications


def createHITType(mturk, titleOfHIT, descriptionOfHIT, payment, durationPerHIT, keywordsForHIT, approvalDelay, qualifications):


    HIT = mturk.register_hit_type(title=titleOfHIT,
                        description=descriptionOfHIT,
                        reward=payment,
                        duration=durationPerHIT,
                        keywords=keywordsForHIT,
                        approval_delay=approvalDelay,
                        qual_req=qualifications)

    return HIT[0].HITTypeId



testbed = LiveOnlineExperiment()
testbed.runOurSystem()
