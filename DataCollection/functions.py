__author__ = 'kgoel93'
from boto.mturk.question import SelectionAnswer, Overview, QuestionForm, Question, QuestionContent, FormattedContent, AnswerSpecification, FreeTextAnswer
from boto.mturk.connection import MTurkConnection, MTurkRequestError
from boto.mturk.qualification import Requirement,Qualifications,PercentAssignmentsApprovedRequirement,NumberHitsApprovedRequirement
import random

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
    question = Question(identifier=identifier,
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


#Both start_index and end_index are included
def getTweets(filename,start_index,end_index):
    Tweets = {}
    GroundTruths = {}

    with open(filename) as f:
        for line in f:
            split_list = line.rstrip().split(",")
            tweet_id,true_label,tweet = int(split_list[0]),split_list[1],",".join(split_list[2:])
            if tweet_id <= end_index and tweet_id >= start_index:
                Tweets[tweet_id] = tweet
                GroundTruths[tweet_id] = true_label

    print "Tweets Loaded."
    return Tweets, GroundTruths

def printAccountBalance(mturk):
    print "Account Balance:"
    print mturk.get_account_balance()


def createQualifications(sandbox):
    qualifications = Qualifications()
    if sandbox:
        MastersQualID = '2F1KVCNHMVHV8E9PBUB2A4J79LU20F'
    else:
        MastersQualID = '2NDP2L92HECWY8NS8H3CK0CP5L9GHO'

    #qualifications.add(Requirement(MastersQualID,'DoesNotExist',required_to_preview=True))
    qualifications.add(PercentAssignmentsApprovedRequirement('GreaterThanOrEqualTo',90,True))
    qualifications.add(NumberHitsApprovedRequirement('GreaterThanOrEqualTo',100,True))

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


### HIT Review & Approval ###
def retrieve_reviewable_hits(mturk, HITFile, HITTypeID):
    """
    Get completed HITs from Mechanical Turk.
    MTurk will only return HITs if all the assignments have
    been completed for a HIT, so there will be no returns
    if 6 assignments out of 10 have been completed, for example.
    """
    f = open(HITFile,'r')
    resultsFile = open('Results','w')
    HITList = []
    for HITID in f:
        if (HITID.rstrip() != 'O' and HITID.rstrip() != 'N' and HITID.rstrip() != 'M'):
            HITList.append(HITID.rstrip())

    for hit in HITList:
        print hit
        resultsFile.write("%s\n" % (hit))
        # A HIT can have multiple assignments
        assignments = mturk.get_assignments(hit,page_size=100)
        for assignment in assignments:
            resultsFile.write("%s|%s" % (assignment.WorkerId,assignment.AssignmentId))
            for answer in assignment.answers[0]:
                resultsFile.write("|%s,%s" %(answer.qid,answer.fields[0]))
            resultsFile.write("\n")
            #try:
            #    mturk.approve_assignment(assignment.AssignmentId)
            #except MTurkRequestError:
            #    pass'''

        #mturk.disable_hit(hit)
        #mturk.expire_hit(hit)
        #mturk.extend_hit(hit,expiration_increment=172800)
        #mturk.change_hit_type_of_hit(hit_id=hit,hit_type=HITTypeID)
    f.close()
    resultsFile.close()
    print "Done with Writing HIT Results"
