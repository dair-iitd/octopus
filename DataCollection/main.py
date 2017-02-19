from functions import *
from boto.mturk.price import Price
from time import sleep
import sys

'''
All Parameter Values
should be entered below.
'''
IS_SANDBOX = False
DEBUG = 1 #2 to print out all requests

#Connect to MTurk
mturk = connectToMTurk(IS_SANDBOX,DEBUG)

#mturk.grant_bonus('A1MZJFYDSRR3OI','3BXQMRHWKZYDBPWA0P550P02ZSFUM6',Price(0.13),"12.5% Bonus for finishing 25 HITs at $0.04 per HIT")
#sleep(10)

printAccountBalance(mturk)

def postHITs():

    titleOfHIT = "Sentiment Analysis for Tweets"
    descriptionOfHIT = "Choose the sentiment (positive/negative) that most closely matches that of the tweet shown. Each HIT contains 10 tweets, and we are "\
                        "posting a total of 40 HITs. Fast approvals (within 24 hours)."
    keywordsForHIT = ["Sentiment", "Sentiment Analysis", "Twitter", "Categorization", "Classification",
                      "Binary Choice", "Multiple Choice", "Twitter Sentiment Analysis"]

    approvalDelay = 86400 #24 Hours

    durationPerHIT = 600 #10 Minutes

    lifetimeOfHIT = 86400 #24 Hours

    #In Dollars
    payment = 0.03

    #Number of Max Assignments
    MaxAssignments = 20

    #Creating separate qualifications
    qualifications = createQualifications(IS_SANDBOX)

    NumberOfHITsToPost = 4

    HIT_TypeID = createHITType(mturk,titleOfHIT,descriptionOfHIT,payment,durationPerHIT,keywordsForHIT,approvalDelay,qualifications)

    # retrieve_reviewable_hits(mturk,'OrdinaryPoolActiveHITs', MasterHITTypeID)
    # exit()
    # sleep(60)

    #Get all tweets
    file_location = '/Users/kgoel93/Desktop/octopus/octopus/DataCollection/square_twitter_data.csv'

    Tweets, GroundTruths = getTweets(file_location,1201,1240)

    QuestionTemplate = "Which of the following choices below best describes the sentiment expressed in the given tweet?"

    overviewTitle = '<b><font color="black" face="Helvetica" size="3"> Instructions </font></b>'
    overviewDescription = '<font color="black" face="Helvetica" size = "2">1. <i>Choose the sentiment (either positive or negative) that <b>best (most closely)</b> matches the ' \
                       "sentiment expressed in the given tweet.</i></font>" , \
                          '<font color="black" face="Helvetica" size = "2">2.<i> There are 10 separate sentences given to you below. The answers for separate sentences are not related. </i></font>', \
                          '<font color="black" face="Helvetica" size = "2">3.<i> Example: "That was the best day ever!" is a positive sentiment, while "I do not like that guy :/" is a negative sentiment. Please use your judgment! </i></font>'


    HIT_IDs = []
    for j in xrange(NumberOfHITsToPost):
        numberOfQuestions = 10
        currentHITToPost = [i for i in xrange(1200 + j*numberOfQuestions + 1,1200+(j+1)*numberOfQuestions + 1)]
        listOfTweets = [Tweets[i] for i in xrange(1200+j*numberOfQuestions + 1,1200+ (j+1)*numberOfQuestions + 1)]

        questionForm = createQuestionForm(overviewTitle,overviewDescription,numberOfQuestions,listOfTweets,list(currentHITToPost))

        HIT = mturk.create_hit(hit_type=HIT_TypeID,lifetime=lifetimeOfHIT,questions=questionForm,max_assignments=MaxAssignments)

        HIT_IDs.append(HIT[0].HITId)

        print "Posted HIT %d" % (j+1)


    #Store the HIT_IDs
    with open('HITS_%d' % (int(payment*100)),'w') as f:
        for hit in HIT_IDs:
            f.write(hit + "\n")


def expireHITs(price):

    with open('HITS_%d' % (int(price*100))) as f:
        for line in f:
            print "Expired HIT " + line.rstrip()
            mturk.expire_hit(line.rstrip())


def getResults(price):

    resultsFile = open('Results_%d' %(int(price*100)),'w')
    with open('HITS_%d' % (int(price*100))) as f:
        for line in f:
            hit = line.rstrip()
            print hit
            assignments = mturk.get_assignments(hit,page_size=100)
            for assignment in assignments:
                resultsFile.write("%s|%s|%s|%s|%s" % (assignment.HITId,assignment.WorkerId,assignment.AssignmentId,assignment.AcceptTime
                    ,assignment.SubmitTime))
                for answer in assignment.answers[0]:
                    resultsFile.write("|%s|%s" % (answer.qid,answer.fields[0]))

                resultsFile.write("\n")

    resultsFile.close()

def getStartTimes(price):

    with open('HITS_%d' % (int(price*100))) as f:
        for line in f:
            hit = line.rstrip()
            print hit
            hit = mturk.get_hit(hit)
            print hit[0].CreationTime
            # print [method for method in dir(object) if callable(getattr(hit, method))]
            # print hit.CreationTime
            # print inspect.getmembers(hit, lambda a:not(inspect.isroutine(a)))

#postHITs()
#expireHITs(0.04)
# getResults(0.03)
getStartTimes(0.06)



