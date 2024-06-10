TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=1 python generate.py --model_path /data2/yyy/LAS-Diffusion/results/withoutPert/3/epoch\=3999.ckpt --generate_method latent_interpolation  --num_generate  11 --steps 100   --input_path1 /data2/yyy/LAS-Diffusion/outputs/withoutPert/3/interp/Ev/homo3d/0binary_voxel  --input_path2  /data2/yyy/LAS-Diffusion/outputs/withoutPert/3/interp/Ev/homo3d/10binary_voxel   --cls True --cls_model_path /data2/yyy/Projects/LAS-Diffusion/regressor_halfunet/bulk_divided_by_Kb/try_copy/ckpts/epoch=156.ckpt



