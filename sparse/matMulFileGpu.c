// This code reads matrices from file and multiplies those matrices


#include<petscmat.h>
#include <petscsys.h>

//reading matrix from the binary file
void readMatrixFromFile(Mat& SysMatrix, const std::string& Filename)
{
    
    PetscPrintf(PETSC_COMM_WORLD,"\n reading global matrix from binary file ... ");

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
  PetscMPIInt rank;
  Mat A; // Sparse matrix object
  Mat B; // Sparse matrix
  Mat C;
  
  PetscLogDouble start_time, end_time;
  PetscLogDouble elapsed_time;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);

  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetType(A, MATAIJCUSPARSE);
  readMatrixFromFile(A, "../data/sparse/Plate_3K/PlateMat_K.mat");

  MatCreate(PETSC_COMM_WORLD, &B);
  MatSetType(B, MATAIJCUSPARSE);
  readMatrixFromFile(B, "../data/sparse/Plate_3K/PlateMat_M.mat");


  PetscPrintf(PETSC_COMM_WORLD, "Starting Matrix Multiplication...\n ");

  PetscTime(&start_time);

  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  
  PetscTime(&end_time);
  
  PetscPrintf(PETSC_COMM_WORLD, "Completed Matrix Multiplication!\n ");

  elapsed_time = end_time - start_time;
  printf("Elapsed time in GPU: %f seconds\n", elapsed_time);

  MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);


  PetscReal norm;
  MatNorm(C,  NORM_1 , &norm);
  printf("\nNorm = %f \n",norm);


  // Destroy the matrix
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

