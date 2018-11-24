import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import OrderedDict 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data = pd.read_csv('project3_dataset1.txt',sep='\t',header=None)
print(data)

labels = data.iloc[:,-1].values
features = data.iloc[:,:-1].values
all_data = data.iloc[:,:].values

print("labels",labels,"\n","features",features)

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    print(counts)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity
	

initialval = gini(all_data)
print("initialval gini ",initialval)

