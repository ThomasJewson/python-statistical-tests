# %%
import pandas as pd
import seaborn as sns

MARJ_IND_LOC = R"https://raw.githubusercontent.com/ThomasJewson/datasets/master/MarijuanaUsePartying/marij1_indiv.csv"
marj_ind = pd.read_csv(MARJ_IND_LOC)
marj_ind
# %%

# %%
marj_ind["party"]
