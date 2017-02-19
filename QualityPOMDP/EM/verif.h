#include "data.h"

extern int* truth;
extern int* sample;
void Load_truth( Dataset *data );
void Print_truth( Dataset *data );
void Accuracy( Dataset *data );
void Randomizer( int );
void Print_Set( int );
double Posterior( Dataset*, int );


