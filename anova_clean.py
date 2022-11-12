# %% 1. Setup
import pandas as pd
import numpy as np
import scipy.stats

"""
Description: Frequency of Marijuana use and Party/Dance Participation
among youths.

Variable Names

marijUse    /*  1=Never, 2= <1/Month, 3= >1/Month, 4= >1/Day  */
party       /* 1=Not at all,  2=Somewhat, 3=A great Deal */
numStdnt
"""

MARJ_IND_LOC = R"https://raw.githubusercontent.com/ThomasJewson/datasets/master/MarijuanaUsePartying/marij1_indiv.csv"
marj_ind = pd.read_csv(MARJ_IND_LOC)

# %% 2. Calculating grand mean, sst and ssw

grand_mean = marj_ind["party"].mean()
marj_ind["squared_error"] = (marj_ind["party"] - grand_mean) ** 2

sst = marj_ind["squared_error"].sum()
sst

group_means = marj_ind.groupby("marijUse")["party"].mean()


def get_variance_within(marij_group: pd.Series):
    return (marij_group - group_means[marij_group.name]) ** 2


marj_ind["variance_within"] = marj_ind.groupby("marijUse")["party"].apply(
    get_variance_within
)
ssw = marj_ind["variance_within"].sum()
ssw

# %% 3. Calculating ssb
ssb = (
    ((marj_ind.groupby("marijUse")["party"].mean() - grand_mean) ** 2)
    * marj_ind.groupby("marijUse")["party"].count()
).sum()
ssb

ssb = sst - ssw

# %% 4. F statistic

num_groups = marj_ind["marijUse"].nunique()
num_observations = len(marj_ind)

dof_within = num_observations - num_groups
dof_between_groups = num_groups - 1

f_statistic = (ssb / dof_between_groups) / (ssw / dof_within)
f_statistic

# %% 5. Hypothesis testing
# H0 = Population means are equal
# H1 = Not all population means are equal

f_critical = scipy.stats.f.ppf(q=1 - 0.05, dfn=dof_between_groups, dfd=dof_within)
f_critical

if f_statistic > f_critical:
    print("Reject H0, thus, not all population means are equal.")
else:
    print("Cannot reject that all population means are equal")


# %% Distribution diagram from Wikipedia
import seaborn as sns

sns.violinplot(x="party", data=marj_ind)

# %%
