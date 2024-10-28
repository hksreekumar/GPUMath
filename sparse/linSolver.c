// Solver for Sparse matrices using MUMPS - LU Decomposition


#include<petscksp.h>
#include<petscsys.h>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    Mat A;
    Vec x, b, u;   // x - solution 
    KSP ksp;    
    PC pc;  //pre-conditioner
    PetscMPIInt rank, size, i, j, matVal;
    PetscInt m = 100, n = 100;


    PetscInitialize(&argc, &argv, NULL, NULL);

    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Create a sparse matrix
    //ierr = MatCreateAIJCUSPARSE(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, 0, NULL, 0, NULL, &A);

    //Another way to create matrix
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
   
    //set the sparse matrices type by passsing arguements 
    //PetscCall(MatSetFromOptions(A));    // -mat_type mpiaijcusparse (or) -mat_type mpiaij 

    //set matrix type
    PetscCall(MatSetType(A, MATMPIAIJCUSPARSE));
    PetscCall(MatSetUp(A));


    //Create vectors
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &u));


    //insert values in matrix
    for (i = 0; i < m; i++) {
    for (j = 0; j < (n/2); j++) {
      matVal = (i + 1) * (j + 1); 
      if(i == j)
     {
      MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
      ierr = MatSetValue(A, i, j, matVal, INSERT_VALUES);
     } 
      }
    }

    // Assemble the matrix
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


    //Insert values into vector
    //PetscCall(VecSetValues(u, vecSize, const PetscInt ix[], const PetscScalar y[], InsertMode iora))

    VecSet(u, 1);

    //print vec
    //VecView(u, PETSC_VIEWER_STDOUT_WORLD);


    // KSP Solver setup
    
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);

  #if defined(PETSC_HAVE_MUMPS)
    if(size > 1) PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  #endif
    PetscCall(KSPSetFromOptions(ksp));

    /* 1. Solve linear system A x = b */
    // b = A u
    PetscCall(MatMult(A, u, b));
    PetscCall(KSPSolve(ksp, b, x));

    //VecView(b, PETSC_VIEWER_STDOUT_WORLD);

    // Destroy the matrix and vectors
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&u));

    PetscFinalize();
    return 0;
}
