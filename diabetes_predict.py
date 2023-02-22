# create function to load a pretrained model.
import pickle
import warnings 
warnings.filterwarnings('ignore')
import argparse


# EXAMPLE USAGE:
# python diabetes_predict.py --pregnancies 6, --glucose 148, --blood_pressure 72, --skin_thickness 35, --insulin 125, --bmi 33.6, --diabetespedigreefunction 0.627, --age 50  

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--pregnancies', required=True, help='how many pregnancies the patient had.')
parser.add_argument('--glucose', required=True, help='patient glucose level')
parser.add_argument('--blood_pressure', required=True, help='patient blood pressure')
parser.add_argument('--skin_thickness', required=True, help='patient skin thickness')
parser.add_argument('--insulin', required=True, help='patient insulin level')
parser.add_argument('--bmi', required=True, help='patient body mass index')
parser.add_argument('--diabetespedigreefunction', required=True, help='patient diabetes pedigree function')
parser.add_argument('--age', required=True, help='patient age')
args=parser.parse_args()


# parse arguments and turn values into float.
pregnancies = int(float(args.pregnancies))
glucose = int(float(args.glucose))
blood_pressure = int(float(args.blood_pressure))
skin_thickness = int(float(args.skin_thickness))
insulin = int(float(args.insulin))
bmi = int(float(args.bmi))
diabetespedigreefunction = int(float(args.diabetespedigreefunction))
age = int(float(args.age))


def load_model(model):
    """
        SUMMARY
            loads model from file
        ARGS
            model: the filename for the model you want to load. 
        RETURNS
            loaded_model (model) loaded model. 
    """
    with open(model, 'rb') as file:
        loaded_model = pickle.load(file)
        return loaded_model
    
# load pretrained model
svm = load_model('svc_model.pkl')


def insert_patient(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age):
    """
        SUMMARY:
            Runs inference on given set of data passed in as arguments.
        ARGS:
            pregnancies
            glucose 
            blood_pressure 
            skin_thickness
            insulin
            bmi
            diabetespedigreefunction, 
            age.
        RETURNS:
            prediction (str) returns a prediction that is either 0 or 1 for no diabetes predicted or diabetes predicted. 
    """
    prediction = svm.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age]]) # where we predict on the data
    if prediction == 0:
        prediction = 'no diabetes predicted'
    elif prediction ==1:
        prediction = 'diabetes predicted'
    return prediction


if __name__=='__main__':
    # EXAMPLE
    # python diabetes_predict.py --pregnancies 6, --glucose 148, --blood_pressure 72, --skin_thickness 35, --insulin 125, --bmi 33.6, --diabetespedigreefunction 0.627, --age 50  
    patient_outcome=insert_patient(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age)
    print(patient_outcome)
