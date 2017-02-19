import heapq

class SimpleSelect(object):

    def __init__(self,qualityPOMDPs):
        self.priorityQueue = []
        for qualityPOMDP in qualityPOMDPs:
            self.addQuestion(qualityPOMDP)

    '''
    Add a question to the priority queue, which is now available to be allocated to an incoming worker.
    '''
    def addQuestion(self,qualityPOMDP):
        phi = self.getPriorityScore(qualityPOMDP)
        heapq.heappush(self.priorityQueue,(-phi,qualityPOMDP.question.question_id))


    '''
    Select a question based on the greedy score. Returns the questionID.
    '''
    def getQuestion(self,debug=False):
        e = heapq.heappop(self.priorityQueue)
        if debug:
            print "Popped question %d with phi=%1.3f." % (e[1],e[0])
        return e[1]


    '''

    '''
    def getPriorityScore(self,qualityPOMDP):
        return qualityPOMDP.estimateBallotsToCompletionUsingFrontierFinding(3,0)
