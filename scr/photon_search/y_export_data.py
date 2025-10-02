# -*- coding: utf-8 -*-

import pandas as pd
from d_decision_tree import DTy_pred
from e_random_forest import RFy_pred
from f_adaboost import ABCy_pred
from c_split_data import y_test

# making a csv file
abc = pd.DataFrame({'true_label': y_test,"pred_label" : ABCy_pred})
dt = pd.DataFrame({'true_label': y_test,"pred_label" : DTy_pred})
rf = pd.DataFrame({'true_label': y_test,"pred_label" : RFy_pred})

abc.to_csv("/home/anna/photon_detection/ML/data/ABC2-007-results.csv", index=False)
dt.to_csv("/home/anna/photon_detection/ML/data/DT2-007-results.csv", index=False)
rf.to_csv("/home/anna/photon_detection/ML/data/RF2-007-results.csv", index=False)
