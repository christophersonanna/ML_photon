# -*- coding: utf-8 -*-

import a_get_data as get_data

# dataset breakdown
Photons = 0
Not_photons = 0
for i in range(0,len(get_data.X_test)):
    if get_data.X_test['photon'][i] == 1:
        Photons += 1
    else:
        Not_photons += 1
        
print('Photons: ', Photons)
print('Not Photons: ', Not_photons)

