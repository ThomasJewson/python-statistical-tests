# %%
import pandas as pd
import numpy as np

#%%
MARJ_IND_LOC = R"https://raw.githubusercontent.com/ThomasJewson/datasets/master/MarijuanaUsePartying/marij1_indiv.csv"

"""
Description: Frequency of Marijuana use and Party/Dance Participation
among youths.

Variable Names

marijUse    /*  1=Never, 2= <1/Month, 3= >1/Month, 4= >1/Day  */
party       /* 1=Not at all,  2=Somewhat, 3=A great Deal */
numStdnt
"""

# %%
# Use one-way ANOVA to test whether going to more partys makes it more
# likely that you will use marj

# one level therefore use one-way

# I am studying what affect going to more partys has on Marj use.

# dependent variable is party
# independent variable is marijUse7

marj_ind = pd.read_csv(MARJ_IND_LOC)

# marj_ind["score"] = marj_ind["marijUse"]
marj_ind["count"] = marj_ind["marijUse"]
marj = marj_ind.groupby(["party", "marijUse"]).agg({"count": np.sum})
marj


def get_score(nums: tuple):
    return nums[0] * nums[1]


marj = marj.reset_index("marijUse")
marj["score"] = marj.apply(get_score, axis=1)
marj_means = marj.groupby(marj.index)["score"].mean()
grand_mean = marj_means.mean()
grand_mean
# %%
(marj_means - grand_mean) ** 2
# %%
marj_ind = pd.read_csv(MARJ_IND_LOC)
marj_means = marj_ind.groupby("marijUse").mean()
grand_mean = marj_means.mean()
grand_mean
# %%
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

ssb = ((
    (marj_ind.groupby("marijUse")["party"].mean() - grand_mean) ** 2
) * marj_ind.groupby("marijUse")["party"].count()).sum()
ssb

ssb = sst - ssw 

# %%
sst - ssw 
