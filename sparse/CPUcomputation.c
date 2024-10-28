#include<petscmat.h>
#include <petscsys.h>
#include<petscksp.h>



//reading matrix from the binary file
void readMatrixFromFile(Mat& SysMatrix, const std::string& Filename)
{

    PetscViewer viewerK;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, Filename.c_str(), FILE_MODE_READ, &viewerK);
    MatLoad(SysMatrix, viewerK);
    PetscViewerDestroy(&viewerK);

    MatAssemblyBegin(SysMatrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(SysMatrix, MAT_FINAL_ASSEMBLY);
    
    PetscInt Istart, Iend;
    MatGetOwnershipRange(SysMatrix, &Istart, &Iend);
    printf("%i: %i - %i\n",PetscGlobalRank, Istart, Iend);
}

void readVectorFromFile(Vec &SysVector, const std::string &Filename)
{

    PetscViewer viewerK;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, Filename.c_str(), FILE_MODE_READ, &viewerK);
    VecLoad(SysVector, viewerK);
    PetscViewerDestroy(&viewerK);

    VecAssemblyBegin(SysVector);
    VecAssemblyEnd(SysVector);
}

void Compute(char *path){

  PetscMPIInt rank, size, sizeFVec;
  Mat m_K, m_M, m_D,KDYN; 
  Vec m_F, m_X;
  KSP ksp;    
  PC pc;  

  PetscLogDouble start_time, end_time, elapsed_time1, elapsed_time2, elapsed_time3, globalStart_time, globalEnd_time, globalElapsed_time;

  PetscInt w, f = 100;
  
  PetscComplex PETSC_i;

  char pathK[50], pathD[50], pathM[50],pathF[50] ;
  strcpy(pathK, path);
  strcpy(pathD, path);
  strcpy(pathM, path);
  strcpy(pathF, path);

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank); 
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  
  // read matrix m_K
  MatCreate(PETSC_COMM_WORLD, &m_K);
  MatSetType(m_K, MATAIJ);
  readMatrixFromFile(m_K,  strcat(pathK, "PlateMat_K.mat"));

  // read matrix m_D
  MatCreate(PETSC_COMM_WORLD, &m_D);
  MatSetType(m_D, MATAIJ);
  readMatrixFromFile(m_D,  strcat(pathD, "PlateMat_D.mat"));

  // read matrix m_M
  MatCreate(PETSC_COMM_WORLD, &m_M);
  MatSetType(m_M, MATAIJ);
  readMatrixFromFile(m_M,  strcat(pathM, "PlateMat_M.mat"));

  //read vector F
  VecCreate(PETSC_COMM_WORLD, &m_F);
  readVectorFromFile(m_F, strcat(pathF,"PlateVec_F.mat"));
  VecGetSize(m_F, &sizeFVec);
  VecZeroEntries(m_F);

  VecSetValue(m_F, 50*5+2, 1., INSERT_VALUES);

  VecAssemblyBegin(m_F);
  VecAssemblyEnd(m_F);
 
  printf("Reading completed\n");

  //setting the values
  w = 2 * 3.14 * 100;

  PetscTime(&globalStart_time);
  for(int i=0; i< 3; i++)
 {
  
  // K, M, D
  // KDYN = -omega*omeaga*M+i*omega*D+K
  // duplicate K -> KDYN
  // Y = a*X+Y -> KDYN =  i*omega*D + KDYN
  // Y = a*X+Y -> KDYN =  -omega*omega*M + KDYN
  // Destroy KDYN

  //Copy matrix K to mat KDYN
  MatDuplicate(m_K, MAT_COPY_VALUES, &KDYN);


  //Duplicate Vec F to Vec X
  VecDuplicate(m_F, &m_X);
  VecZeroEntries(m_X);

  //MatAXPY(Mat Y, PetscScalar a, Mat X, MatStructure str)
  PetscTime(&start_time);
  MatAXPY(KDYN, (PETSC_i * w), m_D, DIFFERENT_NONZERO_PATTERN);
  PetscTime(&end_time);
  
  elapsed_time1 = end_time - start_time;
  // printf("Elapsed time in GPU, Step 1: %f seconds\n", elapsed_time1);

  //Step 2
  PetscTime(&start_time);
  MatAXPY(KDYN,-w*w,m_M, DIFFERENT_NONZERO_PATTERN);
  PetscTime(&end_time);
  elapsed_time2 = end_time - start_time;
  // printf("\nElapsed time in GPU, Step 2: %f seconds\n", elapsed_time2);

  /*PetscReal Xnorm1;
  MatNorm(KDYN,  NORM_1 , &Xnorm1);
  PetscPrintf(PETSC_COMM_WORLD,"\nNorm = %f",Xnorm1);
  exit(-1);*/

  PetscInt M, N;
  MatGetSize(KDYN, &M, &N);
  PetscPrintf(PETSC_COMM_WORLD, "  solving linear system of %d x %d equations ...\n", M,N);

  //Step 3
  // KSP Solver setup for CPU matrices
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPGetPC(ksp, &pc);
  //KSPSetType(ksp, KSPPREONLY);
  PCSetType(pc, PCLU);
  PCFactorSetMatSolverType(pc, MATSOLVERMUMPS);
  KSPSetOperators(ksp, KDYN, KDYN);
  PCFactorSetUpMatSolverType(pc);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);

// Kdyn = x * F, below the output is m_X vector
  PetscTime(&start_time);
  KSPSolve(ksp, m_F, m_X);
  PetscTime(&end_time);
  elapsed_time3 = end_time - start_time;

 
  // printf("\nElapsed time in GPU, Step 3: %f seconds\n", elapsed_time3);
  // printf("\nComputation Completed!\n");
  PetscPrintf(PETSC_COMM_WORLD,"\nTotal time taken to complete all the steps: %f seconds\n", (elapsed_time1 + elapsed_time2 + elapsed_time3));

  //VecView(m_X, PETSC_VIEWER_STDOUT_WORLD);

  PetscReal Xnorm;
  VecNorm(m_X,  NORM_1 , &Xnorm);
  PetscPrintf(PETSC_COMM_WORLD,"\nNorm = %f",Xnorm);
  //exit(-1);

  VecDestroy(&m_X);
  KSPDestroy(&ksp);
  MatDestroy(&KDYN);

 } 

  PetscTime(&globalEnd_time);
  globalElapsed_time = globalEnd_time - globalStart_time;
  PetscPrintf(PETSC_COMM_WORLD,"\nOverall time in CPU: %f seconds\n", globalElapsed_time);

  // Release memory
  VecDestroy(&m_F);
  MatDestroy(&m_M);
  MatDestroy(&m_D);
  MatDestroy(&m_K);

}


int main(int argc, char **argv) 
{

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc, &argv, (char*)0, NULL); if (ierr) return ierr;

  char path[30] = "../data/sparse/Plate_1M/";

  Compute(path);
  
  ierr = PetscFinalize();
  return ierr;
}

