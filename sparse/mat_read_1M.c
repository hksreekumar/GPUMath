#include<petscmat.h>
#include <petscsys.h>
#include <iostream>

void readVectorFromFile(Vec &SysVector, const std::string &Filename)
{
    PetscPrintf(PETSC_COMM_WORLD,"  reading global vector from binary file ... ");

    PetscViewer viewerK;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, Filename.c_str(), FILE_MODE_READ, &viewerK);
    VecLoad(SysVector, viewerK);
    PetscViewerDestroy(&viewerK);

    VecAssemblyBegin(SysVector);
    VecAssemblyEnd(SysVector);

    PetscPrintf(PETSC_COMM_WORLD,"  finished\n");
}


void Mat& readMatrixFromFile(Mat& SysMatrix, const std::string& Filename)
{

    PetscPrintf(PETSC_COMM_WORLD,"  reading global matrix from binary file ... ");

    PetscViewer viewerK;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, Filename.c_str(), FILE_MODE_READ, &viewerK);
    MatLoad(SysMatrix, viewerK);
    PetscViewerDestroy(&viewerK);

    MatAssemblyBegin(SysMatrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(SysMatrix, MAT_FINAL_ASSEMBLY);
    
    PetscPrintf(PETSC_COMM_WORLD,"  finished\n");
    return SysMatrix;
}

int main(int argc, char **argv) 
{
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;

    // read vector
    Vec m_F;
    VecCreate(PETSC_COMM_WORLD, &m_F); 
    readVectorFromFile(m_F, "../data/sparse/Plate_1M/PlateVec_F.mat");
    //VecView(m_F, PETSC_VIEWER_STDOUT_WORLD);
    VecDestroy(&m_F);

    // read matrix
    Mat m_K;
    MatCreate(PETSC_COMM_WORLD, &m_K);
    readMatrixFromFile(m_K, "../data/sparse/Plate_1M/PlateMat_K.mat");
    //MatView(m_K, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&m_K);

    // read matrix
    Mat m_M;
    MatCreate(PETSC_COMM_WORLD, &m_M);
    readMatrixFromFile(m_M, "../data/sparse/Plate_1M/PlateMat_M.mat");
    //MatView(m_M, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&m_M);

    // read matrix
    Mat m_D;
    MatCreate(PETSC_COMM_WORLD, &m_D);
    readMatrixFromFile(m_D, "../data/sparse/Plate_1M/PlateMat_D.mat");
    //MatView(m_D, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&m_D);

    ierr = PetscFinalize();
    return ierr;
}