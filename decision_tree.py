import sys
# Standard libraries
import numpy as np
import pandas as pd

# Matplotlib for visualization
import matplotlib.pyplot as plt

# Scikit-learn for machine learning
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

# Additional libraries (image generator)
import pydotplus
from IPython.display import Image

#LOAD DATA
my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")
my_data.head()

print(my_data.shape)

#PRE-PROCESSING
#Remove the column containing the target name since it doesn't contain numeric values.
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

#LABLE ENCODING
# Sklearn Decision Trees does not handle categorical variables. We can still convert these features to numerical values using the LabelEncoder() method to convert the categorical variable into dummy/indicator variables.

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

y = my_data["Drug"]
y[0:5]


#SETTING UP THE DECISION TREE
#Now train_test_split will return 4 different parameters. We will name them:
# Assuming X and y are your features and target
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, 
# and the random_state ensures that we obtain the same splits.

# MODELING
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print('default parameters ',drugTree)  # it shows the default parameters

# Fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset, y_trainset)

#PREDICTION
# your test data
predTree = drugTree.predict(X_testset)
print('PREDICTION: drug tree -', predTree[:5])

#EVALUATION
print(" EVALUATION:DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#VISUALIZATION


export_graphviz(drugTree, out_file='./tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])



#After executing the code below, a plot  would be generated which contains the decision tree image.

# Assuming drugTree is your trained model
plt.figure(figsize=(15,10))  # Set the figure size
plot_tree(drugTree, filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
plt.show()

