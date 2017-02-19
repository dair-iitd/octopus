import heapq

class FractionalSelect(object):

    def __init__(self,qualityPOMDPs):
        self.priorityQueue = []
        for qualityPOMDP in qualityPOMDPs:
            self.addQuestion(qualityPOMDP)


    '''
    Add a question to the priority queue, which is now available to be allocated to an incoming worker.
    '''
    def addQuestion(self,qualityPOMDP):
        phi = self.getPriorityScore(qualityPOMDP)
        heapq.heappush(self.priorityQueue,(-phi,qualityPOMDP.question.question_id)) #use -phi because min-heap


    '''
    Select a question based on the fractional utility score. Returns the questionID.
    '''
    def getQuestion(self,debug=False):
        e = heapq.heappop(self.priorityQueue)
        if debug:
            print "Popped question %d with phi=%1.3f." % (e[1],e[0])
        return e[1]

    '''
    Computes the fractional utility score for a question, using its Quality POMDP agent.
    We define the fractional utility score, \phi = 1/theta \sum_(i=1 to theta) [v'_max - v_max]/k_i
    where v'_max = max(v'_1,v'_0) beliefs on the answer value after k-steps, for the i_th iteration.
    k_i is the number of ballots taken in the i_th simulation.
    '''
    def getPriorityScore(self,qualityPOMDP,simulations=50,method=2):
        v_max = qualityPOMDP.qualityPOMDPBelief.v_max
        if method == 1:
            priorityScore = 0
            for _ in xrange(simulations):
                v_max_submit_avg,num_ballots = qualityPOMDP.simulatePOMDPUntilSubmit()
                priorityScore += (v_max_submit_avg - v_max)/float(num_ballots)
            priorityScore /= float(simulations)
            return priorityScore
        elif method == 2:
            priority_score = qualityPOMDP.expectedFractionalValue(v_max)
            return priority_score

