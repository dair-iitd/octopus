#include <stdio.h>
#include "verif.h"
#include "prob_functions.h"

int *truth;
int *sample;
double *postZ1;
double *postZ0;

void Load_truth( Dataset *data )
{
  int i, j;
  int idx;
  char str[1024];
  FILE *fp = fopen("d_implied.txt", "rt");
  double priorZ1;
  /* Read parameters */
  int a, b;
  double d;
  truth = (int*) malloc(sizeof(int) * data->numImages);
  for (j = 0; j < data->numImages; j++) {
    fscanf(fp, "%d,%f,%d\n", &a, &d, &b);
    truth[j] = b;
  }
}

void Print_truth( Dataset *data )
{
  int j;

  for (j = 0; j < data->numImages; j++) {
    printf( "%d %d\n", j, truth[j] );
  }
}

void Accuracy( Dataset *data )
{
  int j, total = 0;
  for (j = 0; j < data->numImages; j++) {
    if ( ( data->probZ1[j] > 0.5 && truth[j] == 1 ) || ( data->probZ1[j] <= 0.5 && truth[j] == -1 ) )
      total++;
    else {
      printf( "%d\n", j );
    }
  }
  printf( "Accuracy %f\n", (double)total/data->numImages );
}

void Randomizer( int size )
{
  sample = (int*) malloc (size * sizeof(int));
  int total = 20;
  int perm[20];
  for ( int i = 0; i < total; ++i ) {
    perm[i] = i;
  }

  for ( int i = 0; i < size; ++i ) {
    int remain_num = total - i;
    int p = rand() % remain_num;
    int temp = perm[i];
    perm[i] = perm[i+p];
    sample[i] = perm[i];
    perm[i+p] = temp;
  }

}

void Print_Set( int size )
{
  for ( int i = 0; i < size; ++i ) {
    printf("%d ", sample[i]);
  }
  printf("\n");
}

double Posterior( Dataset *data, int size )
{
  int j, total = 0, idx = 0;
  int images = data->numImages;
  postZ1 = (double*) malloc (images * sizeof(double));
  postZ0 = (double*) malloc (images * sizeof(double));

  // prior
  for ( j = 0; j < images; ++j ) {
    postZ1[j] = 0.5;
    postZ0[j] = 0.5;
  }

  // posterior
  for (idx = 0; idx < data->numLabels; idx++) {
    int i = data->labels[idx].labelerId;
    int k;
    for ( k = 0; k < size; ++k ) {
      if ( i == sample[k] )
        break;
    }
    if ( k == size )
      continue;
    int j = data->labels[idx].imageIdx;
    int lij = data->labels[idx].label;
    postZ1[j] += logProbL(lij, 1, data->alpha[i], data->beta[j]);
    postZ0[j] += logProbL(lij, 0, data->alpha[i], data->beta[j]);
  }

  // validation
  for (j = 0; j < data->numImages; j++) {
    if ( ( postZ1[j] > postZ0[j] && truth[j] == 1 ) || ( postZ1[j] <= postZ0[j] && truth[j] == -1 ) )
      total++;
  }
  double accuracy = (double)total/images;
  //rintf( "Accuracy %f\n", accuracy );

  free(sample);
  free(postZ1);
  free(postZ0);

  return accuracy;
}
