import torch
from pathlib import Path
import numpy as np
import os
import joblib
import pdb

class DatasetforRegressor(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_folder: str, 
                 ):
        super().__init__()
        self.dataset_paths = []
        self.dataset_paths.extend(
            [p for p in Path(f'{dataset_folder}').glob('**/*binary_voxel')])
        print('number of data', ' = ', len(self.dataset_paths))




    def __len__(self):
        return len(self.dataset_paths)

    def __getitem__(self, index):
        dataset_path = self.dataset_paths[index]

        res = {}

        tensor_path = str(dataset_path).replace("binary_voxel", "binary_C")
        with open(tensor_path, 'rb') as f:
            binary_data = np.fromfile(f, dtype=np.float32)

        vol_path = str(dataset_path).replace("binary_voxel", "vol")
        with open(vol_path, 'rb') as f:
            binary_data_vol = np.fromfile(f, dtype=np.float32)


        tensor_feature = - np.ones((10,), dtype=np.float32)  

        tensor_feature[0] = binary_data[0]
        tensor_feature[1] = binary_data[7]
        tensor_feature[2] = binary_data[14]
        tensor_feature[3] = binary_data[21]
        tensor_feature[4] = binary_data[28]
        tensor_feature[5] = binary_data[35]
        tensor_feature[6] = binary_data[1]
        tensor_feature[7] = binary_data[2]
        tensor_feature[8] = binary_data[8]

        E_data = tensor_feature[0:1].reshape(1, -1)
        scaler_E = joblib.load("./scaler_C11")
        E_data_map = scaler_E.transform(E_data).reshape(-1)

        G_data = tensor_feature[3:4].reshape(1, -1)
        scaler_G = joblib.load("./scaler_C44")
        G_data_map = scaler_G.transform(G_data).reshape(-1)

        v_data = tensor_feature[6:7].reshape(1, -1)
        scaler_v = joblib.load("./scaler_C12")
        v_data_map = scaler_v.transform(v_data).reshape(-1)

        tensor_feature = np.array([E_data_map[0], G_data_map[0], v_data_map[0], binary_data_vol[0]], dtype=np.float32)

        res["tensor_feature"] = tensor_feature
        c11 = binary_data[0]
        c12 = binary_data[1]
        c44 = binary_data[21]
        vol = binary_data_vol[0]
        
        bulk = (c11 + 2*c12)/3
        mu=0.3
        G=1e6/(2*(1+mu))
        K=1e6/(3*(1-2*mu))
        Kb=4*G*K*vol/(4*G+3*K*(1-vol))
        res["bulk_Kb"] = (bulk/Kb).astype(np.float32)

        with open(dataset_path, 'rb') as f:
            binary_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
            dim = (64, 64, 64)
            binary_array = np.reshape(binary_data, dim, order="F")
            binary_array = binary_array.astype(dtype=np.float32)

        for i in range(8):
            tmp = binary_array[i * 8:(i + 1) * 8, :, :]
            binary_array[i * 8:(i + 1) * 8, :, :] = tmp[::-1] 

        res["occupancy"] = np.expand_dims(2 * binary_array - 1, axis=0)

        return res
