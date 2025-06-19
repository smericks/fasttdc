import json
import numpy as np

static_dv_files = [('InferenceRuns/exp0_2/static_datavectors_seed'+str(i)+'.json') for i in range(1,11)]

catalog_idxs_list = []

for sdv_file in static_dv_files:

    with open(sdv_file, 'r') as file:
        data_vector_dict_list = json.load(file)
        catalog_idxs = []
        for dv_dict in data_vector_dict_list:
            catalog_idxs.extend(np.asarray(dv_dict['catalog_idxs']))

        catalog_idxs_list.append(catalog_idxs)

np.save('catalog_idxs_per_seed.npy',np.asarray(catalog_idxs_list))