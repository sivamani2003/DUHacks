from flask import Flask, render_template, request
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import pyttsx3
from sklearn import preprocessing
#from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import warnings
import joblib 

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# Load data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

# Group data by prognosis
reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Initialize models
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
model = SVC()
model.fit(x_train, y_train)

# Check if a saved model exists, if not, train and save it
try:
    clf = joblib.load('trained_model.pkl')
except FileNotFoundError:
    # Initialize and train the model
    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    # Save the trained model
    joblib.dump(clf, 'trained_model.pkl')
    
# Feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Initialize dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {}


def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Ensure row has at least two elements
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print("Hello, ", name)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []
    
def calc_condition(symptoms_exp, num_days):
    sum = 0
    for item in symptoms_exp:
        sum += severityDictionary.get(item, 0)

    average_severity = (sum * num_days) / (len(symptoms_exp) + 1)

    if average_severity > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names, symptoms_exp, num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    for node in range(tree_.node_count):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == symptoms_exp:
                val = 1
            else:
                val = 0
            if val <= threshold:
                node = tree_.children_left[node]
            else:
                symptoms_present.append(name)
                node = tree_.children_right[node]
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = input(syms + "? : ")
                if inp.lower() == "yes":
                    symptoms_exp.append(syms)
            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                return present_disease[0], description_list.get(present_disease[0], "No description available")
            else:
                return present_disease[0], description_list.get(present_disease[0], "No description available"), \
                       second_prediction[0], description_list.get(second_prediction[0], "No description available")

    return symptoms_present


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    symptoms_exp = request.form.get('symptoms_exp')
    num_days = int(request.form.get('num_days'))
    result = tree_to_code(clf, cols, symptoms_exp, num_days)
    return render_template('result.html', result=result, num_days=num_days)



if __name__ == '__main__':
    getDescription()
    getSeverityDict()
    getprecautionDict()
# getInfo()
    app.run(debug=True)