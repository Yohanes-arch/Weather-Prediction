import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('weather.csv')
df = df.dropna()
# df.head()

#Dropping the target result column
inputs = df.drop('RainTomorrow', axis='columns')
y = df['RainTomorrow']
RainToday_lab = LabelEncoder()

# Labelling a column and dropping unrequired columns
inputs['Rain_Today'] = RainToday_lab.fit_transform(inputs['RainToday'])
inputs_n = inputs.drop(['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RISK_MM', 'RainToday'], axis='columns')
# inputs_n

# Training the data model
X = inputs_n[['MaxTemp', 'Evaporation', 'WindSpeed9am', 'Humidity9am', 'Pressure9am']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7)

#Invoking decision tree for predicting
clf = tree.DecisionTreeClassifier()
clf.fit(X_test, y_test)
#print(clf.predict(X_test))

#Presenting run with line over only number to make easier to read
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

print(color.BOLD+"\nWelcome to Weather Forecasting by Group 7!"+"\nAccuracy of data train is "+"{} \n".format(clf.score(X_train,y_train))+color.END)

while True:
    inp1 = float(input("The Maximum Temperature     :   "))
    inp2 = float(input("The Evaporation Rate        :   "))
    inp3 = float(input("The Wind Speed at 9am       :   "))
    inp4 = float(input("The Humidity Level at 9 am  :   "))
    inp5 = float(input("The Pressure Level at 9 am  :   "))

    if clf.predict([[inp1,inp2,inp3,inp4,inp5]]) == "No" and inp2 > 0.00001 and inp3 > 0.00001 and inp4 > 0.00001 and inp5 > 0.00001:
        print("\tWill it be rain soon? ")
        print("\t~Oh! The chance to rain soon is " + color.GREEN + color.BOLD +"Less Likely"+ color.END+", so enjoy your day... :)")
        break
    elif clf.predict([[inp1,inp2,inp3,inp4,inp5]]) == "Yes" and inp2 > 0.00001 and inp3 > 0.00001 and inp4 > 0.00001 and inp5 > 0.00001:
        print("\tWill it be rain soon? ")
        print("\t~Uh Oh! Looks like "+color.DARKCYAN+color.BOLD+"it will be rain soon..." +color.END+", brace yourself")
        break
    elif (clf.predict([[inp1,inp2,inp3,inp4,inp5]]) == "No" and inp2 < 0.00001 and inp3 < 0.00001 and inp4 < 0.00001 and inp5 < 0.00001) or(clf.predict([[inp1,inp2,inp3,inp4,inp5]]) == "Yes" and inp2 < 0.00001 and inp3 < 0.00001 and inp4 < 0.00001 and inp5 < 0.00001): 
        print("\tEvap Rate/WindSpeed/HumidLvl/PressLvl cannot be minus! Terminate command...")
        continue
    else:
        print('Bad input value, please re-input again...\n')
        continue

print(clf.score(X_test, y_test))
