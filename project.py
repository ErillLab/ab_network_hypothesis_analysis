import networkx as nx
import numpy as np
import scipy as scp
from collections import defaultdict,Counter
from itertools import combinations
from utils import concat,bs,h,transpose,mi,hamming
from project_utils import random_combination,rpower_law
from tqdm import tqdm,trange
import random
from matplotlib import pyplot as plt
from parse_experimental_data import experimental_data

class NetStruct(object):
    def __init__(self,adjacencies,names=None,connectivity=1):
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

    def clamped_converge(self,state,clamp_settings):
        history = {}
        while not hashify(state) in history:
            new_state = self.iterate(state)
            new_state = clamp(state,clamp_settings)
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

    def sample_from_clamped_equilibrium(self,init_state=None,clamp_settings=[]):
        """run network dynamics until equilibrium, clamping input vertices to desired levels."""
        if init_state is None:
            init_state = np.random.randint(0,2,self.V)
        hist = self.clamped_converge(init_state,clamp_settings)
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
            new_state = clamp(state,experiment)
            new_att = self.converge_to_attractor(new_state)
            att_size = float(len(new_att))
            #print "att_size:",att_size
            for att_state in new_att:
                sanity += prob/att_size
                if agrees_with_experiment(att_state,experiment):
                    lik += prob/att_size
            #print "sanity:",sanity
        return lik

    def experiment_likelihood(self,(treatment,observations),trials=1000):
        """Given experimental data in the form of clamped treatment variables
        and observed response, estimate likelihood of hypothesis."""
        init_obs = {k:v[0] for k,v in observations.items() if sum(v) >= 0} # exclude nans
        final_obs = {k:v[1] for k,v in observations.items() if sum(v) >= 0}
        discrepancies = []
        final_states = []
        final_unclamped_states = []
        desired_states = []
        desired_unclamped_states = []
        diffs = []
        unclamped_diffs = []
        for i in trange(trials):
            rstate = random_state(self.V)
            init_state = clamp(rstate,init_obs)
            final_state = self.sample_from_clamped_equilibrium(init_state,treatment)
            final_unclamped_state = self.sample_from_equilibrium(init_state)
            final_states.append(final_state)
            final_unclamped_states.append(final_unclamped_state)
            desired_state = clamp(final_state,final_obs)
            desired_states.append(desired_state)
            desired_unclamped_state = clamp(final_unclamped_state,final_obs)
            desired_unclamped_states.append(desired_unclamped_state)
            discrepancy = hamming(final_state,desired_state)
            unclamped_discrepancy = hamming(final_unclamped_state,desired_state)
            diff = final_state - desired_state
            unclamped_diff = final_unclamped_state - desired_unclamped_state
            diffs.append(diff)
            unclamped_diffs.append(unclamped_diff)
            discrepancies.append(discrepancy)
        print "distinct final states:",len(set(map(tuple,final_states)))
        print "distinct desired states:",len(set(map(tuple,desired_states)))
        print "distinct final unclamped states:",len(set(map(tuple,final_unclamped_states)))
        print "distinct desired unclamped states:",len(set(map(tuple,desired_unclamped_states)))
        print "distinct diffs:",len(set(map(tuple,diffs)))
        print "distinct unclamped diffs:",len(set(map(tuple,unclamped_diffs)))
        print [i for i,d in enumerate(diffs[0]) if d != 0]
        return discrepancies

    def experiments_likelihood(self,experiments,trials=100):
        return concat([self.experiment_likelihood(experiment,trials)
                       for experiment in experiments])
            

def main_experiment(network,exp_data,hyp_trials = 10,trials_per_exp=100):
    """try to find a good hypothesis which minimizes discrepancy"""
    hyps = []
    discs = []
    for i in range(hyp_trials):
        hyp = Hypothesis(network)
        disc = hyp.experiments_likelihood(exp_data,trials_per_exp)
        print sum(disc)
        hyps.append(hyp)
        discs.append(disc)
    return hyps,discs

def clamp(state,experiment):
    if type(experiment) is dict:
        experiment = clamp_sabot(experiment)
    new_state = np.copy(state)
    for i,val in experiment:
        new_state[i] = val
    return new_state

def clamp_sabot(treatment_dict):
    """convert a dictionary of the form {name:val} into a list of tuples
    that clamp function understands"""
    idx_from_name = {name:i for (i,name) in reduced_network.names.items()} # FIX THIS
    return [(idx_from_name[name],val) for (name,val) in treatment_dict.items()]
    
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
        #print "idx:",idx
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

def big_test(num_experiments=1000):
    V = 2 # number of vertices
    #k = 50 # number of incoming edges per vertex
    #num_experiments = 10 # number of perturbations to perform
    num_inputs = 1 # number of inputs to be clamped (first N in array)
    num_outputs = 1 # number of vertices to be measured
    indegrees = [rpower_law(M=V) for i in range(V)]
    true_tts = [random_truth_table(indeg) for indeg in indegrees] # ground truth for boolean functions at each vertex
    adjs = []
    # initialize adjacency matrix randomly
    for j,indeg in zip(range(V),indegrees):
        # choose incoming edges without replacement
        incoming_edges = random_combination(range(V),indeg)
        for in_edge in incoming_edges:
            adjs.append((in_edge,j,random.choice([1,-1])))
    g = NetStruct(adjs) # build a network structure object out of adjacencies
    true_hyp = Hypothesis(g,true_tts) # build hypothesis out of netstruct + vertex functions
    # do the experiments
    experimental_evidence = []
    for _ in trange(num_experiments):
        #decide how num_inputs (0..num_inputs) will be clamped
        perturbation = [(i,random.randrange(2)) for i in range(num_inputs)]
        # randomize initial state and clamp
        init_state = clamp(random_state(V),perturbation)
        # run state to equilibrium
        final_state = true_hyp.sample_from_equilibrium(init_state)
        # observe outputs (V-num_outputs...V)
        observations = [(i,final_state[i]) for i in range(V-num_outputs,V)] # last num_output states observed
        # record observations
        experimental_evidence.append(perturbation + observations)
    return experimental_evidence
        
def mi_from_experiments(experiments):
    cols = transpose([map(lambda x:x[1],row) for row in experiments])
    plt.imshow([[mi(col1,col2,correct=False) for col1 in cols] for col2 in (cols)],
               interpolation='none')
    plt.colorbar()
    plt.show()
    
def parse_network(fname,connectivity=1):
    """parse network and return graph object"""
    with open(fname) as f:
        raw_lines = [line.strip().split(',') for line in f.readlines()]
    lines = [(source,target,int(sgn)) for (source,sgn,target) in raw_lines]
    all_names = list(set(concat([(source,target) for source,target,sgn in lines])))
    idx_from_name = {name:i for (i,name) in enumerate(all_names)}
    name_from_idx = {i:name for (name,i) in idx_from_name.items()}
    processed_lines = [(idx_from_name[src],idx_from_name[trg],sgn) for (src,trg,sgn) in lines]
    return NetStruct(processed_lines,name_from_idx,connectivity=connectivity)

def parse_ab_network():
    return parse_network("ab_network_structure.csv")

def parse_reduced_network():
    return parse_network("PKN_Liver_SaezRod_2009",connectivity=0.75)
    
ab_network = parse_ab_network()
reduced_network = parse_reduced_network()

def hypothesize(net_struct):
    """given a network structure, return a random hypothesis respecting the network structure"""
    in_degrees = net_struct.graph.in_degree()
    ks = [in_degrees[i] for i in xrange(net_struct.V)]
    return [random_truth_table(k) for k in ks]

