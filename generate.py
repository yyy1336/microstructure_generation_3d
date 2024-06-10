import torch
import numpy as np
from network.model_trainer import DiffusionModel
from utils.mesh_utils import voxel2mesh
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups
import argparse
import os
from tqdm import tqdm
import joblib
import math
from pathlib import Path
import json
import random
import time
from bitstring import BitArray
import pdb

def generate_latent_interpolation(
    model_path: str,
    input_path1: str,
    input_path2: str,
    output_path: str = "./outputs",
    cls_model_path: str = " ",
    ema: bool = True,
    num_generate: int = 36,
    steps: int = 50,
    truncated_time: float = 0.0,
    cls: bool = False
):
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    postfix = f"3/interp/bulk"
    root_dir = os.path.join(output_path, postfix)

    ensure_directory(root_dir)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    with open(input_path1, 'rb') as f:
        binary_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
        dim = (64, 64, 64)
        binary_array = np.reshape(binary_data, dim, order="F")
        binary_array = binary_array.astype(dtype=np.float32)
    for i in range(8):
        tmp = binary_array[i * 8:(i + 1) * 8, :, :]
        binary_array[i * 8:(i + 1) * 8, :, :] = tmp[::-1]
    img1 = torch.from_numpy(np.expand_dims(2 * binary_array - 1, axis=(0, 1))).cuda()

    with open(input_path2, 'rb') as f:
        binary_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
        dim = (64, 64, 64)
        binary_array = np.reshape(binary_data, dim, order="F")
        binary_array = binary_array.astype(dtype=np.float32)
    for i in range(8):
        tmp = binary_array[i * 8:(i + 1) * 8, :, :]
        binary_array[i * 8:(i + 1) * 8, :, :] = tmp[::-1]
    img2 = torch.from_numpy(np.expand_dims(2 * binary_array - 1,  axis=(0, 1))).cuda()

    if(cls):
        vol_path = str(input_path1).replace("binary_voxel", "vol")
        with open(vol_path, 'rb') as f:
            binary_data_vol = np.fromfile(f, dtype=np.float32)
        vol_1 = binary_data_vol[0]

        vol_path = str(input_path2).replace("binary_voxel", "vol")
        with open(vol_path, 'rb') as f:
            binary_data_vol = np.fromfile(f, dtype=np.float32)
        vol_2 = binary_data_vol[0]

        tensors = np.zeros((num_generate, 1), dtype=np.float32)
        for i in range(num_generate):
            theta = i/(num_generate - 1)
            vol = vol_1 * (1 - theta) + vol_2 * theta
            tensors[i, :] = np.array([vol], dtype=np.float32)
        res_tensor = generator.sample_cls_interp(img1, img2, tensors, num_generate, steps=steps, truncated_index=truncated_time, cls_model_path=cls_model_path)
    else:  
        res_tensor = generator.sample_unconditional_interp(img1,img2,num_generate,steps=steps, truncated_index=truncated_time)
    
    for index in tqdm(range(num_generate), desc=f'save results in {root_dir}'):
        voxel = res_tensor[index].squeeze().cpu().numpy()
        ## save to voxel
        bits = BitArray()
        voxel[voxel > 0] = 1
        voxel[voxel < 0] = 0
        flat_voxel = np.ravel(voxel)
        bits_str = np.where(flat_voxel == 1, '0b1', '0b0')
        bits = BitArray(''.join(bits_str))
        reordered_bits = BitArray()
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            reordered_bits.append(byte[::-1])  
        out_filename = os.path.join(os.path.join(root_dir, str(index) + ".voxel"))
        with open(out_filename, 'wb') as f:
            reordered_bits.tofile(f)


def generate_based_on_tensor(
    model_path: str,
    tensor_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 1,
    steps: int = 50,
    truncated_time: float = 0.0,
    w: float = 1.0,
):
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    postfix = f"3/cloak"

    root_dir = os.path.join(output_path, postfix)
    ensure_directory(root_dir)


    with open('./merged_C_circle63_45147.json', 'r') as file:
        data = json.load(file)
    vectors = data['merged_C']
    C11 = np.array(vectors['C11'])  
    C12 = np.array(vectors['C12'])
    C44 = np.array(vectors['C44'])
    # vol = np.array(vectors['vol'])
        
        
    for ii in tqdm(range(0, len(C11)), desc=f'save results in one batch in {root_dir}'):
        E_data = C11[ii].reshape(1, -1)
        scaler_E = joblib.load('./scaler_C11')
        E_data_map = scaler_E.transform(E_data).reshape(-1)

        G_data = C44[ii].reshape(1, -1)
        scaler_G = joblib.load('./scaler_C44')
        G_data_map = scaler_G.transform(G_data).reshape(-1)

        v_data = C12[ii].reshape(1, -1)
        scaler_v = joblib.load('./scaler_C12')
        v_data_map = scaler_v.transform(v_data).reshape(-1)
        
        # tensor_c = np.array([E_data_map[0], G_data_map[0], v_data_map[0], vol[ii]], dtype=np.float32)
        tensor_c = np.array([E_data_map[0], G_data_map[0], v_data_map[0], -1], dtype=np.float32)
        
        generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
        time_start = time.time()
        res_tensor = generator.sample_with_tensor(tensor_c=tensor_c, batch_size=num_generate,
                                                    steps=steps, truncated_index=truncated_time, tensor_w=w)
        time_end = time.time()
        print('generte time cost', time_end - time_start, 's')
        for jj in range(num_generate):
            voxel = res_tensor[jj].squeeze().cpu().numpy()
            voxel[voxel > 0] = 1
            voxel[voxel < 0] = 0
            # ## save to obj
            # try:
            #     mesh = voxel2mesh(voxel)
            #     mesh.export(os.path.join(root_dir, str(ii)+'_'+str(jj) + ".obj"))
            #     # fullvoxel = np.empty((128, 128, 128), dtype=np.float32, order='F')
            #     # fullvoxel[64:128, 64:128, 64:128] = voxel
            #     # fullvoxel[0:64, 64:128, 64:128] = voxel[::-1, :, :]
            #     # fullvoxel[64:128, 64:128, 0:64] = voxel[:, :, ::-1]
            #     # fullvoxel[64:128, 0:64, 64:128] = voxel[:, ::-1, :]
            #     # fullvoxel[0:64, 0:64, 64:128] = voxel[::-1, ::-1, :]
            #     # fullvoxel[0:64, 64:128, 0:64] = voxel[::-1, :, ::-1]
            #     # fullvoxel[64:128, 0:64, 0:64] = voxel[:, ::-1, ::-1]
            #     # fullvoxel[0:64, 0:64, 0:64] = voxel[::-1, ::-1, ::-1]
            #     # meshfull = voxel2mesh(fullvoxel)
            #     # meshfull.export(os.path.join(os.path.join(root_dir, str(ii)+'_'+str(jj) + ".obj")))
            # except Exception as e:
            #     print(str(e))

            ## save to voxel
            flat_voxel = np.ravel(voxel)
            bits_str = np.where(flat_voxel == 1, '0b1', '0b0')
            bits = BitArray(''.join(bits_str))
            reordered_bits = BitArray()
            for i in range(0, len(bits), 8):
                byte = bits[i:i+8]
                reordered_bits.append(byte[::-1])  
            out_filename = os.path.join(os.path.join(root_dir, str(ii)+'_'+str(jj) + ".voxel"))
            with open(out_filename, 'wb') as f:
                reordered_bits.tofile(f)

def generate_unconditional(
    model_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 36,
    steps: int = 100,
    truncated_time: float = 0.0,
):
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    postfix = f"{model_name}_{model_id}_{ema}_{steps}_{truncated_time}"
    root_dir = os.path.join(output_path, postfix)

    ensure_directory(root_dir)
    batches = num_to_groups(num_generate, 50)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    for batch in batches:
        res_tensor = generator.sample_unconditional(
            batch_size=batch, steps=steps, truncated_index=truncated_time)

        for i in tqdm(range(batch), desc=f'save results in one batch in {root_dir}'):
            voxel = res_tensor[i].squeeze().cpu().numpy()

            ## save to voxel
            bits = BitArray()
            voxel[voxel > 0] = 1
            voxel[voxel < 0] = 0
            flat_voxel = np.ravel(voxel)
            bits_str = np.where(flat_voxel == 1, '0b1', '0b0')
            bits = BitArray(''.join(bits_str))
            reordered_bits = BitArray()
            for i in range(0, len(bits), 8):
                byte = bits[i:i+8]
                reordered_bits.append(byte[::-1])  
            out_filename = os.path.join(os.path.join(root_dir, str(index) + ".voxel"))
            with open(out_filename, 'wb') as f:
                reordered_bits.tofile(f)
            index += 1

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='generate_unconditional',
                        help="please choose :\n \
                            1. 'generate_unconditional' \n \
                            2. 'generate_based_on_tensor' \n \
                            3. 'latent_interpolation' \n \ ")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cls_model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="/data2/yyy/LAS-Diffusion/outputs/")
    parser.add_argument("--input_path1", type=str, default="")
    parser.add_argument("--input_path2", type=str, default="")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--tensor_path", type=str, default="binary_C") 
    parser.add_argument("--tensor_w", type=float, default=1.0)
    parser.add_argument("--cls", type=str2bool, default=False)
    parser.add_argument("--verbose", type=str2bool, default=False)

    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)
    if method == "generate_unconditional":
        generate_unconditional(model_path=args.model_path, num_generate=args.num_generate,
                               output_path=args.output_path, ema=args.ema, steps=args.steps,
                               truncated_time=args.truncated_time)
    elif method == "generate_based_on_tensor":
        generate_based_on_tensor(model_path=args.model_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                                 num_generate=args.num_generate, truncated_time=args.truncated_time,
                                 tensor_path=args.tensor_path, w=args.tensor_w)
    elif method == "latent_interpolation":
        generate_latent_interpolation(model_path=args.model_path,
                                      input_path1=args.input_path1, input_path2=args.input_path2,
                                      num_generate=args.num_generate,
                                      output_path=args.output_path, ema=args.ema, 
                                      steps=args.steps,truncated_time=args.truncated_time,
                                      cls=args.cls, cls_model_path=args.cls_model_path)
    else:
        raise NotImplementedError
