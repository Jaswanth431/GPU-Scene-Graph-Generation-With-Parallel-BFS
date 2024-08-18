/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}

__global__ void applyTranformations(int *preOrder, int *size, int *nodeIndex, int **translations, int translationCount, int nodeCount, int * globalPosX, int *globalPosY ){
      int id = blockIdx.x*blockDim.x+ threadIdx.x;
      if(id>=translationCount)return;


      int prNode = translations[id][0];
      int moves[4][2] = {{-1,0}, {1, 0}, {0,-1}, {0,1}};

      int prNodeIdx = nodeIndex[prNode];
      for(int i=0; i<size[prNode]; i++){
        int childNode = preOrder[prNodeIdx + i];
        atomicAdd(&globalPosX[childNode], moves[translations[id][1]][0] *translations[id][2]);
        atomicAdd(&globalPosY[childNode], moves[translations[id][1]][1] *translations[id][2]);
      }

}




__global__ void generateOpacities(int **g_mesh,int * g_frameX,int * g_frameY,int * g_globalCoordinatesX,int * g_globalCoordinatesY,int *g_globalOp, int *gFinalPng, int *g_opacity,int  nodeCount, int frameSizeX, int frameSizeY){
      int id = blockIdx.x*blockDim.x+ threadIdx.x;
      if(id>=nodeCount)return;

      int startX = g_globalCoordinatesX[id];
      int endX = startX + g_frameX[id] ;
  
      int startY = g_globalCoordinatesY[id];
      int endY = startY + g_frameY[id] ;

      for(int i=startX; i<endX; i++){
        for(int j=startY; j<endY; j++){
          if(i<0 || j<0 || i>=frameSizeX || j>=frameSizeY)continue;

          int pos =  i*frameSizeY+j;
          atomicMax(&g_globalOp[pos], g_opacity[id]);
        }
      }
}
 
__global__ void generateFinalMatrix(int *nodeOpacityMap,int **g_mesh,int * g_frameX,int * g_frameY,int * g_globalCoordinatesX,int * g_globalCoordinatesY,int *g_globalOp, int *gFinalPng, int *g_opacity,int  nodeCount, int frameSizeX, int frameSizeY){
      int id = blockIdx.x*blockDim.x+ threadIdx.x;
      if(id>=frameSizeX * frameSizeY)return;
      int posX = id/frameSizeY;
      int posY = id%frameSizeY;

      if(g_globalOp[id] == -1){
        gFinalPng[id] = 0;
        return;
      }
      int mshId = nodeOpacityMap[g_globalOp[id]];


      int startX = g_globalCoordinatesX[mshId];
      int startY = g_globalCoordinatesY[mshId];

      int mCurrX = posX - startX;
      int mCurrY = posY - startY;

      int mshSizeY = g_frameY[mshId];
      int mshPos = mCurrX * mshSizeY + mCurrY;
      gFinalPng[id] = g_mesh[mshId][mshPos];
}

int generatePreOrder(int nodeCount, int *preOrder, int *size, int *nodeIndex, int &currCount, int *hOffset, int *hCsr, bool *visited, int currNode){
  preOrder[currCount] = currNode;
  nodeIndex[currNode] = currCount;
  visited[currNode] = true;
  int childCount = 0;
  currCount++;
  for(int i=hOffset[currNode]; i<hOffset[currNode+1]; i++){
     int nxtNode = hCsr[i];
     if(visited[nxtNode])continue;
     childCount+= generatePreOrder(nodeCount, preOrder, size, nodeIndex, currCount, hOffset, hCsr, visited, nxtNode);
  }
  childCount++;
  size[currNode] = childCount;
  return childCount;
}  


int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.


  int nodeCount = scene->getNumNodes(); 
  int *preOrder = (int*) malloc (sizeof (int) *nodeCount) ;
  int *size = (int*) malloc (sizeof (int) *nodeCount) ;
  int *nodeIndex = (int*) malloc (sizeof (int) *nodeCount) ;
  bool *visited = (bool*) malloc (sizeof (bool) *nodeCount) ;
  int currCount = 0;

  
  generatePreOrder(nodeCount, preOrder, size, nodeIndex, currCount, hOffset, hCsr, visited, 0);


  //copy preorder to gpu
  int *g_preorder;
  cudaMalloc(&g_preorder, (sizeof (int) * nodeCount) );
  cudaMemcpy(g_preorder, preOrder, sizeof(int)*nodeCount, cudaMemcpyHostToDevice);

  int *g_size;
  cudaMalloc(&g_size, (sizeof (int) * nodeCount) );
  cudaMemcpy(g_size, size, sizeof(int)*nodeCount, cudaMemcpyHostToDevice);
 
  int *g_nodeIndex;
  cudaMalloc(&g_nodeIndex, (sizeof (int) * nodeCount) );
  cudaMemcpy(g_nodeIndex, nodeIndex, sizeof(int)*nodeCount, cudaMemcpyHostToDevice);
  
  //Apply all the translations in parallel
  //copy translations to GPU
  int **g_translations;
  cudaMalloc(&g_translations, numTranslations * sizeof(int *));
  for (int i = 0; i < numTranslations; i++) {
      int *temp;
      cudaMalloc(&temp, 3 * sizeof(int));
      cudaMemcpy(temp, translations[i].data(), 3 * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(g_translations+i, &temp, sizeof(int*), cudaMemcpyHostToDevice);
  }

  //copy global coordinates
  int *g_globalCoordinatesX, *g_globalCoordinatesY;
  cudaMalloc(&g_globalCoordinatesX, nodeCount*sizeof(int));
  cudaMemcpy(g_globalCoordinatesX, hGlobalCoordinatesX, nodeCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&g_globalCoordinatesY, nodeCount*sizeof(int));
  cudaMemcpy(g_globalCoordinatesY, hGlobalCoordinatesY, nodeCount * sizeof(int), cudaMemcpyHostToDevice);


  int threadsPerBlock = 1024;
  int gpuBlocks = (numTranslations+threadsPerBlock-1)/threadsPerBlock;
  applyTranformations<<<gpuBlocks, threadsPerBlock>>>(g_preorder, g_size, g_nodeIndex, g_translations, numTranslations, nodeCount, g_globalCoordinatesX,g_globalCoordinatesY);  
  cudaDeviceSynchronize();
  //Finally generate the scene matrix in parallel
  // copy opacity to gpu
  int *g_opacity;
  cudaMalloc(&g_opacity, nodeCount*sizeof(int));
  cudaMemcpy(g_opacity, hOpacity, nodeCount * sizeof(int), cudaMemcpyHostToDevice);

  //generate and copy global opacity matrix gpu
  int *g_globalOp;
  cudaMalloc(&g_globalOp, (sizeof (int) * frameSizeX * frameSizeY) );
  cudaMemset(g_globalOp, -1,(sizeof (int) * frameSizeX * frameSizeY));


  //create final matrix in gpu
  int *gFinalPng;
  cudaMalloc(&gFinalPng, (sizeof (int) * frameSizeX * frameSizeY));
  cudaMemcpy(gFinalPng, hFinalPng, sizeof (int) * frameSizeX * frameSizeY, cudaMemcpyHostToDevice);

  //create and copy frame sizes to GPU
  int *g_frameX, *g_frameY;
  cudaMalloc(&g_frameX, nodeCount*sizeof(int));
  cudaMemcpy(g_frameX, hFrameSizeX, nodeCount * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&g_frameY, nodeCount*sizeof(int));
  cudaMemcpy(g_frameY, hFrameSizeY, nodeCount * sizeof(int), cudaMemcpyHostToDevice);

  //copy hMesh to gpu
  int **g_mesh;
  cudaMalloc(&g_mesh, nodeCount*sizeof(int *));
  for (int i = 0; i < nodeCount; i++) {
      int *temp;
      cudaMalloc(&temp, hFrameSizeX[i]*hFrameSizeY[i]* sizeof(int));
      cudaMemcpy(temp, hMesh[i], hFrameSizeX[i]*hFrameSizeY[i] * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(g_mesh+i, &temp, sizeof(int*), cudaMemcpyHostToDevice);
  }

 
  //generate final matrix
  gpuBlocks = (nodeCount+1023)/1024;
  generateOpacities<<<gpuBlocks, 1024>>>(g_mesh, g_frameX, g_frameY, g_globalCoordinatesX, g_globalCoordinatesY,g_globalOp,gFinalPng, g_opacity, nodeCount, frameSizeX, frameSizeY);
  // generateFinalMatrix<<<1, 1>>>(g_mesh, g_frameX, g_frameY, g_globalCoordinatesX, g_globalCoordinatesY,g_globalLock,g_globalOp,gFinalPng, g_opacity, nodeCount, frameSizeX, frameSizeY);

  int maxOpacity = 0;
  for(int i=0;i<nodeCount; i++){
    maxOpacity = max(maxOpacity, hOpacity[i]);
  }

  //generate a map for opacity & node mapping
  int *nodeOpacityMap = (int *)malloc(sizeof(int)*(maxOpacity + 10));

  for(int i=0;i<nodeCount; i++){
    int currOp = hOpacity[i];
    nodeOpacityMap[currOp] = i;
  }

  int *g_nodeOpacityMap ;
  cudaMalloc(&g_nodeOpacityMap, sizeof(int)*(maxOpacity + 10));
  cudaMemcpy(g_nodeOpacityMap, nodeOpacityMap, sizeof(int)*(maxOpacity + 10), cudaMemcpyHostToDevice);


  int totalCells = frameSizeX * frameSizeY;
  gpuBlocks = (totalCells + threadsPerBlock -1)/threadsPerBlock;
  generateFinalMatrix<<<gpuBlocks, threadsPerBlock>>>(g_nodeOpacityMap,g_mesh, g_frameX, g_frameY, g_globalCoordinatesX, g_globalCoordinatesY,g_globalOp,gFinalPng, g_opacity, nodeCount, frameSizeX, frameSizeY);

  cudaMemcpy(hFinalPng, gFinalPng, sizeof (int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);

  // for(int i=0;i<frameSizeX; i++){
  //   for(int j=0;j<frameSizeY; j++){
  //     printf("%d ", hFinalPng[i*frameSizeY+j]);
  //   }
  //   printf("\n");
  // }
  // freeing cuda memory
  cudaFree(g_frameX);
  cudaFree(g_frameY);
  cudaFree(g_globalCoordinatesX);
  cudaFree(g_globalCoordinatesY);
  cudaFree(g_globalOp);
  cudaFree(g_mesh);
  cudaFree(g_opacity);
  cudaFree(g_translations);
  cudaFree(g_preorder);
  cudaFree(g_size);
  cudaFree(g_nodeIndex);
  cudaFree(g_nodeOpacityMap);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
