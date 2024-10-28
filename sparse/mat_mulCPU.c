#include<petscmat.h>
#include <petscsys.h>

int main(int argc, char **argv) 
{

  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt m = 30000, n = 30000; // Size of the matrix
  Mat A; // Sparse matrix object
  Mat B; // Sparse matrix
  Mat C;
  PetscInt i, j;
  PetscScalar value;
  PetscLogDouble start_time, end_time;
  PetscLogDouble elapsed_time;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);

   //MatCreateAIJ - for Parallel CPU matrix

  ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, 0, NULL, 0, NULL, &A);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, 0, NULL, 0, NULL, &B);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, 0, NULL, 0, NULL, &C);

  //Inserting values into the matrix A and B

  for (i = 0; i < m; i++) {
    for (j = 0; j < 10000; j++) {
      value = (i + 1) * (j + 1); 
      if(i == j)
     {
      
      MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
      ierr = MatSetValue(A, i, j, value, INSERT_VALUES);

      MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
      ierr = MatSetValue(B, i, j, value, INSERT_VALUES);
     } 
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

  

  // PetscPrintf(PETSC_COMM_WORLD, "Printing the matrix\n ");
  // MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  // MatView(B, PETSC_VIEWER_STDOUT_WORLD);
  // MatView(C, PETSC_VIEWER_STDOUT_WORLD);
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

