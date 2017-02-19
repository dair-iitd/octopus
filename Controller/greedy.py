import heapq


class GreedySelect(object):

    def __init__(self,k,qualityPOMDPs,cache=0,bundle=None):
        self.priorityQueue = []
        self.lookahead = k
        self.cached = False
        if cache == 1:
            self.cached = True
            self.cache = {}
            possibleV,alpha,beta,qualityPOMDP = bundle
            for v in possibleV:
                qualityPOMDP.qualityPOMDPBelief.setBelief((v + 1)/2.0,alpha,beta)
                self.cache[round((v + 1)/2.0,3)] = self.getPriorityScore(qualityPOMDP,cached=0)

        self.addQuestions(qualityPOMDPs)

    '''
    Add a question to the priority queue, which is now available to be allocated to an incoming worker.
    '''
    def addQuestion(self,qualityPOMDP,cached=None):
        phi = self.getPriorityScore(qualityPOMDP,cached=cached)
        heapq.heappush(self.priorityQueue,(-phi,qualityPOMDP.question.question_id)) #use -phi because min-heap


    '''
    Select a question based on the greedy score. Returns the questionID.
    '''
    def getQuestion(self,debug=False):
        e = heapq.heappop(self.priorityQueue)
        if debug:
            print "Popped question %d with phi=%1.3f." % (e[1],e[0])
        return e[1]


    '''
    Computes the k-step lookahead greedy score for a question, using its Quality POMDP agent.
    We define the k-step lookahead greedy score, \phi = v'_max - v_max
    where v'_max = max(v'_1,v'_0), beliefs on the answer value after k-steps.
    '''
    def getPriorityScore(self,qualityPOMDP,cached=None): #FIXED WITH DEEPCOPY
        if cached is None and self.cached:
            phi = self.cache[round(qualityPOMDP.qualityPOMDPBelief.v_max,3)]
        else:
            phi = qualityPOMDP.expectedValue(self.lookahead) - qualityPOMDP.qualityPOMDPBelief.v_max
        return phi


    def addQuestions(self,qualityPOMDPs,cached=None):
        for qualityPOMDP in qualityPOMDPs:
            self.priorityQueue.append((-self.getPriorityScore(qualityPOMDP,cached),qualityPOMDP.question.question_id))
        heapq.heapify(self.priorityQueue)


    def reset(self):
        self.priorityQueue = []