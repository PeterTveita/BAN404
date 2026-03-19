
# Read libraries
import pandas as pd
import matplotlib.pyplot as plt

# (a) Read the data file
college = pd.read_csv("College.csv")

# (b) Look at data and change name of first column
#     and alternatively let the first column become rownames
college.head()
# Let first column become rownames only
college2 = pd.read_csv('College.csv', index_col =0)
college2.head()
# Change name of first column and let it become rownames
college3 = college.rename ({'Unnamed: 0': 'College '},axis =1)
college3.head()
college3 = college3.set_index('College ')
college3.head()
# Change name of the data frame
college = college3

# (c) Produce a numerical summary of the data frame
college.describe()

# (d) Plot three different columns as a scatterplot matrix
pd.plotting.scatter_matrix(college[["Top10perc", "Apps", "Enroll"]])

# (e) Plot a boxplots for Apps splitted into Private and Public colleges
fig , ax = plt.subplots(figsize =(8, 8))
college.boxplot('Apps', by = 'Private', ax=ax)

# (f) Make a new variable in the dataframe with value 'Yes' if Top10Perc>50 and 'No otherwise'
college['Elite'] = pd.cut(college['Top10perc'],[0,50,100],labels =['No', 'Yes'])
# Tabulate the values of the new variable
college['Elite'].value_counts()
# Boxplots
fig, ax = plt.subplots(figsize = (8,8))
college.boxplot('Outstate', by = 'Elite', ax = ax)

# (g) Produce 4 histograms of Outstate with different number of bins
data = college.Outstate # Another way to extract a variable from a data frame
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
bin_settings = [5, 10, 20, 50]
for ax, bins in zip(axs.flat, bin_settings):
 ax.hist(data, bins=bins, color='skyblue', edgecolor='black')
 ax.set_title(f'Histogram with {bins} bins')

# (h) Plot Outstate against a few other variables
college.columns
fig, axs = plt.subplots(2,2)
for ax, var in zip(axs.flat,["Top10perc", "Apps", "Enroll","S.F.Ratio"]):
    ax.scatter(college[var],college['Outstate'])
    ax.set_xlabel(var)
    ax.set_ylabel('Outstate')

