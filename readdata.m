%% read voxel
fid = fopen(['D:\TO\AIGC4Materials\supp_wucha\Cxx-vol\homo3d\0_0binary_voxel'],'rb'); 
if fid == -1
error('无法打开文件。');
end
dim = [64, 64, 64];
binary_data = fread(fid, prod(dim), 'ubit1');
fclose(fid);
binary_array = reshape(binary_data, dim);
fullvoxel = zeros(128,128,128);
fullvoxel(1:64,1:64,1:64) = binary_array;
fullvoxel(65:128,1:64,1:64) = binary_array(end:-1:1,:,:);
fullvoxel(1:64,65:128,1:64) = binary_array(:,end:-1:1,:);
fullvoxel(1:64,1:64,65:128) = binary_array(:,:,end:-1:1);
fullvoxel(65:128,65:128,1:64) = binary_array(end:-1:1,end:-1:1,:);
fullvoxel(65:128,1:64,65:128) = binary_array(end:-1:1,:,end:-1:1);
fullvoxel(1:64,65:128,65:128) = binary_array(:,end:-1:1,end:-1:1);
fullvoxel(65:128,65:128,65:128) = binary_array(end:-1:1,end:-1:1,end:-1:1);
volumeViewer(fullvoxel)
%% read elasticity tensor
fileID = fopen('D:\TO\AIGC4Materials\dataset\1-90000\1-90000\1-90000-1w\45147binary_C');
C = fread(fileID,[6,6],'float');
fclose(fileID);
c11=C(1,1);
c12=C(1,2);
c44=C(4,4);
%% read volume fraction
fileID = fopen('D:\TO\AIGC4Materials\dataset\1-90000\1-90000\1-90000-1w\45147vol');
vol = fread(fileID,[1,1],'float');
