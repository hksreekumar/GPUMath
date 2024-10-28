#include<petscmat.h>
#include <petscsys.h>

int main(int argc, char **argv) 
{

  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt m = 100, n = 100; // Size of the matrix
  Mat A; 
  Mat B;
  Mat C;
  PetscInt i, j;
  PetscScalar value;
  PetscLogDouble start_time, end_time;
  double elapsed_time;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);

  //MatCreateDenseCUDA - for GPU

  ierr =  MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,m,n, NULL, &A);
  ierr =  MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,m,n, NULL, &B);
  ierr =  MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,m,n, NULL, &C);

  //Inserting values into the matrix A and B

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      value = i * j;
      MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
      ierr = MatSetValue(A, i, j, value, INSERT_VALUES);

      MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
      ierr = MatSetValue(B, i, j, value, INSERT_VALUES);
     } 
    }
  PetscPrintf(PETSC_COMM_WORLD, "Sparse Matrix ready\n"); 

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

  PetscPrintf(PETSC_COMM_WORLD, "Starting Matrix Multiplication...\n ");

  PetscTime(&start_time);

  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  
  PetscTime(&end_time);
  
  PetscPrintf(PETSC_COMM_WORLD, "Completed Matrix Multiplication!\n ");

  elapsed_time = end_time - start_time;
  printf("Elapsed time in CPU: %f seconds\n", elapsed_time);

  MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);


  // Destroy the matrix
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

