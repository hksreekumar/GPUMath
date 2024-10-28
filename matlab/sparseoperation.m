clear all;

KMAT= PetscBinaryRead('../data/sparse/Plate_3K/PlateMat_K.mat');
MMAT= PetscBinaryRead('../data/sparse/Plate_3K/PlateMat_M.mat');
DMAT= PetscBinaryRead('../data/sparse/Plate_3K/PlateMat_D.mat');

omega = 2*pi*100;

KDYN1 = -omega*omega*MMAT + 1i*omega*DMAT + KMAT*(1+0.001*1j);

KDYN = H5CSRToSparseMatFOM('../data/sparse/Plate_3K/eGenSystem_Plate_3K_Exp.hdf5','/Stiffness');

n = size(KMAT,1);
FVEC = zeros(n,1);
FVEC(1) = 1;


x= KDYN\FVEC;