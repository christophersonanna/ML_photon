# -*- coding: utf-8 -*-

import a_get_data as get_data

# Split dataset into training set and test set
y_train = get_data.X_train['photon']
y_test = get_data.X_test['photon']

del(get_data.X_test['photon'])
del(get_data.X_train['photon'])
