import networkx as nx
import numpy as np
import scipy as scp
from collections import defaultdict,Counter
from utils import concat,bs,h
from tqdm import tqdm,trange
import random

class NetStruct(object):
    def __init__(self,adjacencies,names=None):
        """adjacencies describes signed directed graph, i.e. edge i -> j
        encoded as (i,j,1), i -| j encoded as (i,j,-1) names is a
        dictionary of the form {i:i_name}.
        """
        
        self.V = max(concat([[i,j] for (i,j,sgn) in adjacencies])) + 1
        self.adjs = adjacencies
        self.names = names
        self.graph = nx.DiGraph([(src,trg) for (src,trg,sgn) in self.adjs])
        #self.mat = scp.sparse.dok_matrix((self.V,self.V))
        self.mat = np.zeros((self.V,self.V))
        for (src,trg,sgn) in self.adjs:
            self.mat[src,trg] = sgn
        

    def plot(self):
        nx.draw_graphviz(self.graph,prog='dot',labels=self.names)
        
class Hypothesis(object):
    def __init__(self,graph,tts=None):
        self.graph = graph
        if tts is None:
            tts = hypothesize(graph)
        self.tts = tts
        self.V = self.graph.V
        
    def iterate(self,state):
        input_sets = [[] for j in xrange(self.graph.V)]
        for i,j,sgn in self.graph.adjs:
            # input set for j consists of list [(i,signed state[i])]
            input_sets[j].append((i,state[i] if sgn > 0 else 1-state[i]))
        new_state = np.zeros(len(state),dtype=np.int)
        for j,(input_set,tt) in enumerate(zip(input_sets,self.tts)):
            #print j,(input_set,tt)
            sorted_input_set = sorted(input_set,key=lambda(idx,in_set):idx)
            inputs = [val for (i,val) in sorted_input_set]
            if inputs:
                new_state[j] = tt(*inputs)
            else:
                new_state[j] = state[j]
        return new_state

    def converge(self,state):
        history = {}
        while not hashify(state) in history:
            new_state = self.iterate(state)
            history[hashify(state)] = hashify(new_state)
            state = new_state
        return history

    def converge_to_attractor(self,state=None):
        if state is None:
            state = random_state(self.V)
        hist = self.converge(state)
        attractor = attractor_from_history(hist)
        return [unhashify(state) for state in attractor]
        

    def sample_from_equilibrium(self,init_state=None):
        """sample from attractors"""
        if init_state is None:
            init_state = np.random.randint(0,2,self.V)
        hist = self.converge(init_state)
        attractor = attractor_from_history(hist)
        hashed_choice = random.choice(attractor)
        return unhashify(hashed_choice)
        
    def analyze_basins(self,N):
        """sample N random states and keep track of how many basins they fall into"""
        attractors = defaultdict(int)
        for trial in trange(N):
            init_state = np.random.randint(0,2,self.V)
            hist = self.converge(init_state)
            att = attractor_from_history(hist)
            attractors[att] += 1
        return attractors

    def likelihood(self,experiment,trials=1000):
        """Given experiment consisting of partial observation, estimate
        likelihood P(D|H).  Experiment is a partial observation of the form [(i,{0,1})]"""
        points = concat([self.converge_to_attractor() for i in xrange(trials)])
        probs = {k:v/float(len(points)) for k,v in Counter(map(hashify,points)).items()}
        #print sum(probs.values())
        #print "distinct states:",len(probs)
        lik = 0
        sanity = 0
        for hashed_state,prob in probs.items():
            state = unhashify(hashed_state)
            new_state = intervene(state,experiment)
            new_att = self.converge_to_attractor(new_state)
            att_size = float(len(new_att))
            #print "att_size:",att_size
            for att_state in new_att:
                sanity += prob/att_size
                if agrees_with_experiment(att_state,experiment):
                    lik += prob/att_size
            #print "sanity:",sanity
        return lik
        
def intervene(state,experiment):
    new_state = np.copy(state)
    for i,val in experiment:
        new_state[i] = val
    return new_state

def agrees_with_experiment(state,experiment):
    return all([state[i] == val for (i,val) in experiment])
    
def random_state(V):
    return np.random.randint(0,2,V)
    
def hashify(state):
    return "".join(map(str,state))

def unhashify(hashed_state):
    return np.array([int(c) for c in hashed_state])

def test_hashification():
    for i in range(100):
        state = random_state(100)
        assert((state == unhashify(hashify(state))).all())
    print "passed"
    
def random_truth_table(k):
    """construct random boolean function from {and, or} on k inputs"""
    # should properly sample uniformly at random from constructible truth tables?
    if k == 0:
        return None
    vs = ["v%s" % i for i in range(k)]
    random.shuffle(vs)
    res_string = reduce(lambda x,y:"(%s) %s %s" % (x,random.choice(["and","or"]),y),vs)
    f_string = "lambda " + ",".join(vs) + ":" + res_string
    f = eval(f_string)
    return f

def attractor_from_history(hist):
    """given a history, extract the attractor.  return it as tuple with
    lexicographically lowest state first"""
    seen_so_far = []
    state = random.choice(hist.keys())
    while state not in seen_so_far:
        seen_so_far.append(state)
        state = hist[state]
    # after loop, state has been seen so far hence lies in the attractor
    cycle = []
    while not state in cycle:
        cycle.append(state)
        state = hist[state]
    try:
        idx,_ = min(enumerate(cycle),key=lambda (i,x):int(x))
        print "idx:",idx
        sorted_cycle = tuple(cycle[idx:]+cycle[:idx])
    except:
        print hist
    assert len(cycle) == len(sorted_cycle) and set(cycle) == set(sorted_cycle)
    return sorted_cycle
    

    
def test_iteration():
    V = 100
    k = 3
    tts = [random_truth_table(3) for i in range(V)]
    adjs = []
    for j in range(V):
        for i in range(k):
            adjs.append((random.randrange(V),j,random.choice([1,-1])))
    g = NetStruct(adjs)
    hyp = Hypothesis(g,tts)
    init_state = np.random.randint(0,2,V)
    return hyp.iterate(init_state)
    
def pop_estimator(obs):
    """Given a vector of observed species counts obs,
    estimate total number of species by bs of shannon entropy"""
    N = float(sum(obs))
    sample = concat([[i for _ in range(v)] for i,v in enumerate(obs)])
    def resample_pop():
        re_obs = Counter(bs(sample)).values()
        return 2**h([v/N for v in re_obs])
    return [resample_pop() for i in range(100)]

def big_test():
    V = 100
    k = 3
    num_experiments = 10
    num_inputs = 4
    num_outputs = 4
    true_tts = [random_truth_table(k) for i in range(V)]
    adjs = []
    for j in range(V):
        for i in range(k):
            adjs.append((random.randrange(V),j,random.choice([1,-1])))
    g = NetStruct(adjs)
    true_hyp = Hypothesis(g,true_tts)
    experimental_evidence = []
    for _ in xrange(num_experiments):
        perturbation = [(i,random.randrange(2)) for i in range(num_inputs)]
        init_state = intervene(random_state(V),perturbation)
        final_state = true_hyp.sample_from_equilibrium(init_state)
        observations = [(i,final_state[i]) for i in range(V-num_outputs,V)] # last num_output states observed
        experimental_evidence.append(perturbation + observations)
    return experimental_evidence
        
    
def parse_ab_network():
    """parse ab network and return graph object"""
    with open("ab_network_structure.csv") as f:
        raw_lines = [line.strip().split(',') for line in f.readlines()]
    lines = [(source,target,int(sgn)) for (source,sgn,target) in raw_lines]
    all_names = list(set(concat([(source,target) for source,target,sgn in lines])))
    idx_from_name = {name:i for (i,name) in enumerate(all_names)}
    name_from_idx = {i:name for (name,i) in idx_from_name.items()}
    processed_lines = [(idx_from_name[src],idx_from_name[trg],sgn) for (src,trg,sgn) in lines]
    return NetStruct(processed_lines,name_from_idx)
    
def hypothesize(net_struct):
    """given a network structure, return a random hypothesis respecting the network structure"""
    in_degrees = net_struct.graph.in_degree()
    ks = [in_degrees[i] for i in xrange(net_struct.V)]
    return [random_truth_table(k) for k in ks]
