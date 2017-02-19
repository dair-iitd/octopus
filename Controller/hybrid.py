import heapq

class HybridSelect(object):

    def __init__(self,qualityPOMDPs):
        self.priorityQueue = [(-10000,qualityPOMDP.question.question_id) for qualityPOMDP in qualityPOMDPs]

    '''
    Add a question to the priority queue, which is now available to be allocated to an incoming worker.
    '''
    def addQuestion(self,qualityPOMDP):
        phi = self.getPriorityScore(qualityPOMDP)
        heapq.heappush(self.priorityQueue,(phi,qualityPOMDP.question.question_id))


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
        return max(qualityPOMDP.getBeliefInAnswer())
