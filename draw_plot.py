import sys
import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

class ClassicalEpsilonGreedy:
  pass

class EpsilonGreedy:
  pass

class ClassicalBUCB:
  pass

class BUCB:
  pass

class UCB:
  pass

class ClassicalUCB:
  pass

class ClassicalThompson:
  pass

class Thompson:
  pass

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

filename=sys.argv[1]

df=pd.read_csv(filename)

#plt.suptitle(sys.argv[1])

plt.figure(figsize=(10,6))

ax=plt.subplot(2,2,1)
plot_df=mean_and_CI(df,ClassicalEpsilonGreedy)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,EpsilonGreedy)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("Epsilon-Greedy")
ax.set_xlabel("Iterations")
ax.set_ylabel("Regret")
ax.legend(loc='upper right')


ax=plt.subplot(2,2,2)
plot_df=mean_and_CI(df,ClassicalBUCB)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,BUCB)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("Beta-UCB")
ax.set_xlabel("Iterations")
ax.set_ylabel("Regret")
ax.legend(loc='upper right')

ax=plt.subplot(2,2,3)
plot_df=mean_and_CI(df,ClassicalUCB)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,UCB)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("UCB")
ax.set_xlabel("Iterations")
ax.set_ylabel("Regret")
ax.legend(loc='upper right')

ax=plt.subplot(2,2,4)
plot_df=mean_and_CI(df,ClassicalThompson)
plot_axes(ax,plot_df,"red","Classic")
plot_df=mean_and_CI(df,Thompson)
plot_axes(ax,plot_df,"blue","Delegation")
ax.set_title("Thompson Sampling")
ax.set_xlabel("Iterations")
ax.set_ylabel("Regret")
ax.legend(loc='upper right')


plt.tight_layout()
plt.savefig(f"{sys.argv[1][:-4]}.pdf",format="pdf",bbox_inches="tight")
plt.show()

