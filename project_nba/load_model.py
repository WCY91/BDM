import os
import pickle
from model import dnn,transformer

def load_decision_tree_model():
    with open("weights/decision_tree_model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

def load_svc_model():
    with open("weights/svm_model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

def load_knn_model():
    with open("weights/knn_model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

def load_dnn_pretrained_model(num_classes,target,res):
    if target == 'money' and res == False:
        model = dnn.classification_model(num_classes)
        model.load_weights("weights/money_baseline_weights.h5")
        return model
    elif target == 'money' and res == True:
        model = dnn.classification_model(num_classes)
        model.load_weights("weights/res_money_baseline_weights.h5")
        return model
    elif target == 'time' and res == False:
        model = dnn.classification_model(num_classes)
        model.load_weights("weights/time_baseline_weights.h5")
        return model
    elif target == 'time' and res == True:
        model = dnn.classification_model(num_classes)
        model.load_weights("weights/res_time_baseline_weights.h5")
        return model

def load_transformer_pretrained_model(num_classes,target):
    if target == 'reg_time':
        model = transformer.build_regression_model()
        model.load_weights('weights/time_reg_transformer.h5')
        return model
    elif target == 'clf_time':
        model = transformer.build_classify_model(num_classes)
        model.load_weights('weights/time_clf_transformer.h5')
        return model
