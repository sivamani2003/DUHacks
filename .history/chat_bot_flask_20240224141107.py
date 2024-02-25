
from flask import Flask, render_template, request
import re
import pandas as pd
import numpy as np
import csv
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
import joblib

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# Load data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Group data by prognosis
reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Check if a saved model exists
try:
    clf = joblib.load('trained_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("The trained model file does not exist. Please train the model first.")

# Initialize dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {}

# Function to read text aloud
def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

# Function to get symptom descriptions
def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        description_list = {row[0]: row[1] for row in csv_reader}

# Function to get symptom severity
def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        severityDictionary = {row[0]: int(row[1]) for row in csv_reader if len(row) >= 2}

# Function to get precautionary measures
def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        precautionDictionary = {row[0]: [row[1], row[2], row[3], row[4]] for row in csv_reader}

# Function to check pattern in input
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

# Function to calculate condition based on symptoms and number of days
def calc_condition(symptoms_exp, num_days):
    sum = 0
    for item in symptoms_exp:
        sum += severityDictionary.get(item, 0)

    average_severity = (sum * num_days) / (len(symptoms_exp) + 1)

    if average_severity > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."

# Function to predict secondary diagnosis based on symptoms
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

# Function to print disease based on node
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

# Function to convert decision tree to code
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

# Flask routes
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
    app.run(debug=True)
