import networkx as nx
import random
from scipy.stats import beta
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time

import ray
ray.init()

def make_graph(num_agents,seed=None):
    """makes the graph. the nodes named {n}_exec are the "execution nodes and have an associated likelihood of successful task execution, denoted p_succ. Also returns the "best" node."""
    #g=nx.generators.fast_gnp_random_graph(num_agents,p=0.3,directed=True,seed=seed)
    g=nx.generators.scale_free_graph(num_agents,seed=seed)
    random.seed(seed)
    best=None
    num_nodes=len(g.nodes)
    for n in range(num_nodes):
        g.add_edge(n,f"{n}_exec")
        g.nodes[f"{n}_exec"]["p_succ"]=random.random()
        if best==None or g.nodes[f"{n}_exec"]["p_succ"]>g.nodes[best]["p_succ"]:
            best=f"{n}_exec"
    return g,g.nodes[best]


@ray.remote
def run_iterations(g,best_node,start_node,num_iters,method_class):
    regret=[]
    payoff=[]

    obj=method_class(g)

    for i in range(num_iters):
        visited=obj.select_node(start_node,[])
        result=random.random()<g.nodes[visited[-1]]["p_succ"]
        obj.update(visited,result)
        regret.append(g.nodes[visited[-1]]["p_succ"]-best_node["p_succ"])
        payoff.append(1) if result else payoff.append(0)
    return (regret,payoff)


class GetNode:
    """This is the class which all methods will inherit from"""
    def __init__(self,graph):
        self.graph=graph
    
    def reset(self):
        pass

    def update(self,visited,result):
        pass
    
    def select_node(node,visited):
        """selects a child node of node which is not in visited. Returns the full path of visited nodes, so the node for execution is visited[-1]"""    
        pass

########################Non-delegation Epsilon greedy#############
EPSILON=0.05
class ClassicalEpsilonGreedy(GetNode):
    def __init__(self,g):
        super().__init__(g)
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1]
    
    def update(self,visited,result):
        for v in visited:
            [s,f]=self.success_fail_node_map[v]
            if result:
                s+=1
            else:
                f+=1
            self.success_fail_node_map[v]=[s,f]

    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        if random.random()<EPSILON:
            choice=random.choice(list(candidates))
        else:
            best_u=0
            for n in candidates:
                if self.success_fail_node_map[n][0]/(self.success_fail_node_map[n][0]+self.success_fail_node_map[n][1]+1)>=best_u:
                    choice=n
                    best_u=self.success_fail_node_map[n][0]/(self.success_fail_node_map[n][0]+self.success_fail_node_map[n][1]+1)
        return self.select_node(choice,visited+[choice])

########################Non-delegation BUCB#############
BUCB_CONST=3
class ClassicalBUCB(GetNode):
    def __init__(self,g):
        super().__init__(g)
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1] #start at 1,1 to get beta to work

    def update(self,visited,result):
        for v in visited:
            [s,f]=self.success_fail_node_map[v]
            if result:
                s+=1
            else:
                f+=1
            self.success_fail_node_map[v]=[s,f]
    
    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        best_u=0

        for n in candidates:
            [s,f]=self.success_fail_node_map[n]
            sr=s/(s+f)+BUCB_CONST*beta.std(s,f)
            if sr>best_u:
                best_u=sr
                choice=n
        return self.select_node(choice,visited+[choice])
########################Non-delegation UCB#############
UCB_CONST=3
class ClassicalUCB(GetNode):
    def __init__(self,g):
        super().__init__(g)
        self.tot_iters=1
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1] #start at 1,1 to get beta to work

    def update(self,visited,result):
        self.tot_iters+=1
        for v in visited:
            [s,f]=self.success_fail_node_map[v]
            if result:
                s+=1
            else:
                f+=1
            self.success_fail_node_map[v]=[s,f]
    
    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        best_u=0

        for n in candidates:
            [s,f]=self.success_fail_node_map[n]
            ucb_term=s/(s+f)+UCB_CONST*math.sqrt(2*math.log(self.tot_iters)/(s+f))
            if ucb_term>best_u:
                best_u=ucb_term
                choice=n
        return self.select_node(choice,visited+[choice])
########################Non-delegation Thompson#############
class ClassicalThompson(GetNode):
    def __init__(self,g):
        super().__init__(g)
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1] #start at 1,1 to get beta to work

    def update(self,visited,result):
        for v in visited:
            [s,f]=self.success_fail_node_map[v]
            if result:
                s+=1
            else:
                f+=1
            self.success_fail_node_map[v]=[s,f]
    
    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        best_u=0

        for n in candidates:
            [s,f]=self.success_fail_node_map[n]
            cur = np.random.beta(s,f)
            if cur>best_u:
                best_u=cur
                choice=n
        return self.select_node(choice,visited+[choice])

########################################################
########################Delegation Epsilon greedy#############
EPSILON=0.05
class EpsilonGreedy(GetNode):
    def __init__(self,g):
        super().__init__(g)
        self.cache={}
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1]
    
    def update(self,visited,result):
        self.cache={}
        #for v in visited:
        [s,f]=self.success_fail_node_map[visited[-1]]
        if result:
                s+=1
        else:
                f+=1
        self.success_fail_node_map[visited[-1]]=[s,f]
    
    def calculate_success(self,node,visited):
        """returns the likelihood of success of node"""
        if node in self.cache:
            return self.cache[node]
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            [s,f]=self.success_fail_node_map[node]
            self.cache[node]=s/(s+f)
            return s/(s+f)

        best=0
        eg=0
        for c in candidates:
            s=self.calculate_success(c,visited+[node])
            if s>best:
                best=s
            eg+=s  #takes into account the EPSILON chance of picking this node
        #total chance is eg + (1-EPSILON)*best
        self.cache[node]=EPSILON*eg/len(candidates)+(1-EPSILON)*best
        return self.cache[node]

    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        best_u=0
        if random.random()<EPSILON:
            choice=random.choice(list(candidates))
        else:
            for c in candidates:
                u=self.calculate_success(c,visited+[c])
                if u>best_u:
                    choice=c
                    best_u=u

        return self.select_node(choice,visited+[choice])
    
########################Delegation Bayesian UCB#############
class BUCB(GetNode):
    def __init__(self,g):
        super().__init__(g)
        self.cache={}
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1]
    
    def update(self,visited,result):
        self.cache={}
        #for v in visited:
        [s,f]=self.success_fail_node_map[visited[-1]]
        if result:
                s+=1
        else:
                f+=1
        self.success_fail_node_map[visited[-1]]=[s,f]
    
    def calculate_success(self,node,visited):
        """returns the likelihood of success of node"""
        if node in self.cache:
            return self.cache[node]
        candidates=set(self.graph.neighbors(node))-set(visited)

        if len(candidates)==0:
            [s,f]=self.success_fail_node_map[node]
            self.cache[node]=s/(s+f)+BUCB_CONST*beta.std(s,f)
            return self.cache[node]

        best=0
        for c in candidates:
            s=self.calculate_success(c,visited+[c])
            if s>=best:
                best=s
        self.cache[node]=best
        return self.cache[node]

    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        best_u=0
        for c in candidates:
            u=self.calculate_success(c,visited+[c]) ####changed from node
            if u>best_u:
                choice=c
                best_u=u

        return self.select_node(choice,visited+[choice])

########################Delegation Standard UCB#############
class UCB(GetNode):
    def __init__(self,g):
        super().__init__(g)
        self.cache={}
        self.tot_iters=1
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1]
    
    def update(self,visited,result):
        self.cache={}
        self.tot_iters+=1
        #for v in visited:
        [s,f]=self.success_fail_node_map[visited[-1]]
        for n in self.graph.nodes:
            if str(n)[-1]=="c":
                [sn,fn]=self.success_fail_node_map[n]
        if result:
                s+=1
        else:
                f+=1
        self.success_fail_node_map[visited[-1]]=[s,f]
    
    def calculate_success(self,node,visited):
        """returns the likelihood of success of node"""
        if node in self.cache:
            return self.cache[node]
        
        candidates=set(self.graph.neighbors(node))-set(visited)

        if len(candidates)==0:
            [s,f]=self.success_fail_node_map[node]
            self.cache[node]=s/(s+f) + UCB_CONST*math.sqrt(2*math.log(self.tot_iters)/(s+f))
            return self.cache[node]

        best=0
        for c in candidates:
            s=self.calculate_success(c,visited+[c])
            if s>=best:
                best=s
        self.cache[node]=best
        return self.cache[node]

    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        best_u=0

        for c in candidates:
            u=self.calculate_success(c,visited+[c])
            if u>=best_u:
                choice=c
                best_u=u

        return self.select_node(choice,visited+[choice])

########################Delegation Thompson#############
class Thompson(GetNode):

    def __init__(self,g):
        super().__init__(g)
        self.cache={}
        self.success_fail_node_map={}
        for n in g.nodes:
            self.success_fail_node_map[n]=[1,1]
    
    def update(self,visited,result):
        self.cache={}
        [s,f]=self.success_fail_node_map[visited[-1]]
        if result:
                s+=1
        else:
                f+=1
        self.success_fail_node_map[visited[-1]]=[s,f]
    
    def calculate_success(self,node,visited):
        """returns the likelihood of success of node"""
        if node in self.cache:
            return self.cache[node]
        
        candidates=set(self.graph.neighbors(node))-set(visited)

        if len(candidates)==0:
            [s,f]=self.success_fail_node_map[node]
            self.cache[node]=np.random.beta(s,f)
            return self.cache[node]
            #return np.random.beta(s,f)

        best=0
        for c in candidates:
            s=self.calculate_success(c,visited+[c])
            if s>best:
                best=s
        self.cache[node]=best
        #return best
        return self.cache[node]

    def select_node(self,node,visited):
        candidates=set(self.graph.neighbors(node))-set(visited)
        if len(candidates)==0:
            return visited
        
        choice=None
        best_u=0
        for c in candidates:
            u=self.calculate_success(c,visited+[c])
            if u>best_u:
                choice=c
                best_u=u

        return self.select_node(choice,visited+[choice])

########################################################

def cum_reg(r):
    s=0
    n=[]
    for a in r:
        s+=a
        n.append(s)
    return n

def update_df(df,method,run,r):
    tdf=pd.DataFrame(cum_reg(r),columns=["Regret"])
    tdf["Method"]=method.__name__
    tdf["Run"]=run
    tdf["Iteration"]=tdf.index
    return pd.concat([df,tdf])

def mean_and_CI(df,method):
    """returns a tuple (mean,(lowerCI,upperCI)) for each iteration"""
    ret_df=pd.DataFrame()
    mn=method.__name__

    ret_df["mean"]=df[df["Method"]==mn].groupby("Iteration")["Regret"].mean()
    ret_df["lowCI"]=df[df["Method"]==mn].groupby("Iteration")["Regret"].apply(lambda x: st.t.interval(0.95,len(x)-1,loc=np.mean(x),scale=st.sem(x))[0])
    ret_df["highCI"]=df[df["Method"]==mn].groupby("Iteration")["Regret"].apply(lambda x: st.t.interval(0.95,len(x)-1,loc=np.mean(x),scale=st.sem(x))[1])
    return ret_df

def plot_axes(ax,plot_df,color,label):
    ax.plot(plot_df["mean"],color=color,label=label)
    ax.fill_between(plot_df.index,plot_df["lowCI"],
                plot_df["highCI"],
                color=color,
                alpha=0.1)



NUM_RUNS=100
NUM_ITERS=20000

NUM_AGENTS=50


df=pd.DataFrame(columns=["Method","Run","Iteration","Regret"])

seed=int(time.time())

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,ClassicalEpsilonGreedy) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,ClassicalEpsilonGreedy,i,a[i][0])

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,EpsilonGreedy) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,EpsilonGreedy,i,a[i][0])

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,ClassicalUCB) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,ClassicalUCB,i,a[i][0])

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,UCB) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,UCB,i,a[i][0])

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,ClassicalBUCB) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,ClassicalBUCB,i,a[i][0])

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,BUCB) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,BUCB,i,a[i][0])

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,ClassicalThompson) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,ClassicalThompson,i,a[i][0])

futures=[run_iterations.remote(make_graph(NUM_AGENTS,seed=seed+i)[0],make_graph(NUM_AGENTS,seed=seed+i)[1],0,NUM_ITERS,Thompson) for i in range(NUM_RUNS)]
a=ray.get(futures)
for i in range(len(a)):
    df=update_df(df,Thompson,i,a[i][0])

    
df.to_csv(f"output_a{NUM_AGENTS}_i{NUM_ITERS}_r{NUM_RUNS}_b{BUCB_CONST}_u{UCB_CONST}_small_world.csv",index=False)

ax=plt.subplot(2,2,1)
plot_df=mean_and_CI(df,ClassicalEpsilonGreedy)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,EpsilonGreedy)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("Epsilon")
ax.legend(loc='upper right')

ax=plt.subplot(2,2,2)
plot_df=mean_and_CI(df,ClassicalBUCB)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,BUCB)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("BUCB")
ax.legend(loc='upper right')

ax=plt.subplot(2,2,3)
plot_df=mean_and_CI(df,ClassicalUCB)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,UCB)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("UCB")
ax.legend(loc='upper right')

ax=plt.subplot(2,2,4)
plot_df=mean_and_CI(df,ClassicalThompson)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,Thompson)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("TS")
ax.legend(loc='upper right')

plt.show()
