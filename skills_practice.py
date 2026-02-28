# %%
import pandas as pd
import numpy as np
import sklearn as sk
# %%
# load data
salary_data = pd.read_csv("/workspaces/week_6/2025_salaries.csv", header=1, encoding="latin-1")
stats = pd.read_csv("/workspaces/week_6/nba_2025.txt", sep=",", encoding="latin-1")
# %%
# check data
salary_data.head()
stats.head()
# %%
# merge the two dataframes on the "Player" column
#help(pd.merge)
merged_data = pd.merge(salary_data, stats, on="Player")
merged_data.head()
# %%
# duplicates in the "Player" column
duplicates = merged_data[merged_data.duplicated(subset="Player", keep=False)]
duplicates.head()
# %%
# Sklearn steps
# 1. Create an instance of the model. ex: mymodel = KMeans(n_clusters=3)
# 2. Fit the model to the data. ex: mymodel.fit(X)
# 3. Make predictions using the model. ex: predictions = mymodel.predict(X)
# 4. Evaluate the model's performance. ex: score = mymodel.score(X)

# For kmeans you don't need to predict, you can just use the labels_ attribute
# to get the cluster assignments for each data point after fitting the model.

# For the lab: make a graph so that shape=cluster, color=salary
# Consider using point score, minutes played, total rebounds, etc to measure performance