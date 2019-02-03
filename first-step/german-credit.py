


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# this uses the german credit data set
# its a quick intro to get our feet wet with python
# we will fit a logistic regression and neural network
# note: the original data is encoded pretty crapily
# each column has its own special codes and the name aren't included
# encodings & column description are on the website:
#  https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
# but i didn't feel like going through and redoing the dat set
#  in a way that makes sense
# especially since we aren't really doing an analysis of the data 
#  but just comparing simple model fits
# this data is the german.data-numeric version
# the names are just a,b,c,... except for bad_loan
file_path = "/Users/andrewbates/Desktop/neural-nets/data/german-credit.csv"

credit = pd.read_csv(file_path, sep = ",")


credit.shape
credit.head()
credit.describe()
credit.columns

# update: changed start of x to 0 b/c python is 0-indexed!
x = credit.iloc[:,0:25]
y = credit.iloc[:, 25]

y.unique()

# barplot of bad_loan (if the loan is bad or not)
# ~ 300 good and ~700 bad
# this seems a bit weird b/c usually # of "bad" is smaller
# maybe the encoding mentioned at 
#  https://www4.stat.ncsu.edu/~boos/var.select/german.credit.html
#  isn't correct?
sns.countplot(x = credit["bad_loan"], data = credit)

# side-by-side bar chart of variable 'a' and good/vs bad loan
pd.crosstab(credit.a,credit.bad_loan).plot(kind='bar')


# ----- create train/test split and standardize -------

x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y,
                                                    test_size = 0.25,
                                                    random_state = 42)

# standardize
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)

# update: this was added after working on concrete and realizing a mistake
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# ----- logistic regression --------------

# note: by default an L2 penalty is used
# https://scikit-learn.org/stable/modules/generated/
#  sklearn.linear_model.LogisticRegression.html
# also: specify a solver or get a warning!
logistic = LogisticRegression(solver = 'lbfgs')
logistic.fit(x_train, y_train)
logistic_pred = logistic.predict(x_test)
logistic_confusion = metrics.confusion_matrix(y_test, logistic_pred)
logistic_confusion # this is just too good!

# array([[ 72,   0],
#       [  0, 178]])



# turn confusion matrix into percent
# givng raw counts isn't very helpful
logistic_confusion / logistic_confusion.astype(np.float).sum(axis=1)

# classification metrics
print("Accuracy:",metrics.accuracy_score(y_test, logistic_pred)) 
print("Precision:",metrics.precision_score(y_test, logistic_pred)) 
print("Recall:",metrics.recall_score(y_test, logistic_pred))


# apparently, the predict() method gives class predictions, not probabilities
# this is a bit weird b/c the logistic response is P(y = 1 | x)
# maybe this is to conform with other classifiers?
# let's get probability estimates and evaluate
logistic_prob_pred = logistic.predict_proba(x_test)[::,1]

# ROC - how is this possible?!
fpr, tpr, thresh = metrics.roc_curve(y_test, logistic_prob_pred,
                                     pos_label = 1)
plt.plot(fpr, tpr)

metrics.roc_auc_score(y_test, logistic_prob_pred)
 

# ----- neural network -----

mlp = MLPClassifier(hidden_layer_sizes = (10, 10), max_iter = 500)
mlp.fit(x_train, y_train)

mlp_pred = mlp.predict(x_test)
metrics.confusion_matrix(y_test, mlp_pred) 
# again, too good. this doesn't seem right

# array([[ 71,   1],
#       [  0, 178]])


mlp_prob_pred = mlp.predict_proba(x_test)[::,1]
metrics.roc_auc_score(y_test, mlp_prob_pred)









