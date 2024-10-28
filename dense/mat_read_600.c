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


void readMatrixFromFile(Mat& SysMatrix, const std::string& Filename)
{

    PetscPrintf(PETSC_COMM_WORLD,"  reading global matrix from binary file ... ");

    PetscViewer viewerK;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, Filename.c_str(), FILE_MODE_READ, &viewerK);
    MatLoad(SysMatrix, viewerK);
    PetscViewerDestroy(&viewerK);

    MatAssemblyBegin(SysMatrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(SysMatrix, MAT_FINAL_ASSEMBLY);

    PetscPrintf(PETSC_COMM_WORLD,"  finished\n");
}

int main(int argc, char **argv) 
{
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;

    // read matrix
    Mat m_K;
    MatCreate(PETSC_COMM_WORLD, &m_K);
    readMatrixFromFile(m_K, "../data/dense/Plate_50K_HO/PlateMat_K.mat");
    //MatView(m_K, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&m_K);

    // read matrix
    Mat m_M;
    MatCreate(PETSC_COMM_WORLD, &m_M);
    readMatrixFromFile(m_M, "../data/dense/Plate_50K_HO/PlateMat_M.mat");
    //MatView(m_M, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&m_M);

    // read matrix
    Mat m_D;
    MatCreate(PETSC_COMM_WORLD, &m_D);
    readMatrixFromFile(m_D, "../data/dense/Plate_50K_HO/PlateMat_D.mat");
    //MatView(m_D, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&m_D);
    
    // read matrix
    Mat m_B;
    MatCreate(PETSC_COMM_WORLD, &m_B);
    readMatrixFromFile(m_B, "../data/dense/Plate_50K_HO/PlateMat_B.mat");
    //MatView(m_B, PETSC_VIEWER_STDOUT_WORLD);
    PetscInt m, n;
    MatGetSizes(m_B, &m, &n);
    

    MatDestroy(&m_B);
    
    // read matrix
    Mat m_C;
    MatCreate(PETSC_COMM_WORLD, &m_C);
    readMatrixFromFile(m_C, "../data/dense/Plate_50K_HO/PlateMat_C.mat");
    //MatView(m_C, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&m_C);

    ierr = PetscFinalize();
    return ierr;
}