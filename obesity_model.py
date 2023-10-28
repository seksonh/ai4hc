import pandas as pd
Obesity  = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Questions of the survey used for initial recollection of information.
# https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub
df = Obesity.copy()
target = 'NObeyesdad'
encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6}
def target_encode(val):
    return target_mapper[val]

df['NObeyesdad'] = df['NObeyesdad'].apply(target_encode)

# Separating X and y
X = df.drop('NObeyesdad', axis=1)
Y = df['NObeyesdad']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('Obesity_clf.pkl', 'wb'))