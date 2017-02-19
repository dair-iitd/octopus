#include <gsl/gsl_vector.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "verif.h"
#include "data.h"
#include "prob_functions.h"

void EM (Dataset *data)
{
	int i, j;
	const double THRESHOLD = 1E-10;
	double Q, lastQ;

	srand(time(NULL));
	
	/* Initialize parameters to starting values */
	for (i = 0; i < data->numLabelers * data->numWorkflows; i++) {
		data->alpha[i] = data->priorAlpha[i];
		/*data->alpha[i] = (double) rand() / RAND_MAX * 4 - 2;*/
	}
	for (j = 0; j < data->numImages * data->numWorkflows; j++) {
		data->beta[j] = data->priorBeta[j];
		/*data->beta[j] = (double) rand() / RAND_MAX * 3;*/
	}
	
	Q = 0;
	EStep(data);
	Q = computeQ(data);
	/*printf("Q = %f\n", Q);*/
  int iter = 0;
	do {
  printf("Iteration %d ", iter++);
		lastQ = Q;
  printf("old Q: %f", lastQ);

		/* Re-estimate P(Z|L,alpha,beta) */
		EStep(data);
		Q = computeQ(data);
		/*printf("Q = %f; L = %f\n", Q, 0.0 computeLikelihood(data));*/
  printf("\tafter E: %f", Q);

		/*outputResults(data);*/
		MStep(data);
		
		Q = computeQ(data);
  printf("\tafter M: %f\n", Q);
		/*printf("Q = %f; L = %f\n", Q, 0.0 computeLikelihood(data));*/
	} while (fabs((Q - lastQ)/lastQ) > THRESHOLD);
	 outputResults(data); 
}

int main (int argc, char *argv[])
{
	Dataset data;

	if (argc < 2) {
		fprintf(stdout, "Usage: em <data>\n");
		fprintf(stdout, "where the specified data file is formatted as described in the README file.\n");
		exit(1);
	}

	//These used to be 1
	double prioralpha = 0.5;
	double priorbeta = 0.5;
	//double prioralpha = 1.0;
	//double priorbeta = 1.0;
  if ( argc > 2 )
    prioralpha = atof(argv[2]);
  if ( argc > 3 )
    priorbeta = atof(argv[3]);
  //printf("BLAH\n");
	readData(argv[1], &data, prioralpha, priorbeta);
	//	printf("LOADING TRUTH\n");
	//Load_truth(&data);
  //printf("STARTING EM\n");
	EM(&data);
	/*
  Accuracy(&data);
  int size;
  int s, samples = 50000;
  for ( size = 3; size <= 11; size+=2 ) {
    double acc = 0;
    for ( s = 0; s < samples; ++s ) {
      Randomizer( size);
      acc += Posterior( &data, size );
    }
    acc /= samples;
    printf("%d %f\n", size, acc);
  }

	free(data.priorAlpha);
	free(data.priorBeta);
	free(data.labels);
	free(data.alpha);
	free(data.beta);
	free(data.probZ1);
	free(data.probZ0);
	*/
}
