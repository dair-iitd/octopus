#include <math.h>
#include <string.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf_erf.h>
#include "prob_functions.h"
#include "data.h"

static const double ALPHA_PRIOR_MEAN = 1, BETA_PRIOR_MEAN = 1;

double my_f (const gsl_vector *x, void *params)
{
	Dataset *data = (Dataset *) params;

	unpackX(x, data);

	return - computeQ(data);
}

void my_df (const gsl_vector *x, void *params, gsl_vector *g)
{
	int i, j;

	Dataset *data = (Dataset *) params;
	double *dQdAlpha = (double *) malloc(sizeof(double) * data->numLabelers * data->numWorkflows);
	double *dQdBeta = (double *) malloc(sizeof(double) * data->numImages * data->numWorkflows);
	
	unpackX(x, data);
	
	gradientQ(data, dQdAlpha, dQdBeta);

	/* Pack dQdAlpha and dQdBeta into gsl_vector */
	for (i = 0; i < data->numLabelers * data->numWorkflows; i++) {
		gsl_vector_set(g, i, - dQdAlpha[i]);  /* Flip the sign since we want to minimize */
	}
	for (j = 0; j < data->numImages * data->numWorkflows; j++) {
	  gsl_vector_set(g, (data->numLabelers * data->numWorkflows) + j, - dQdBeta[j]);  /* Flip the sign since we want to minimize */
	}

	free(dQdAlpha);
	free(dQdBeta);
}

void my_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *g)
{
	*f = my_f(x, params);
	my_df(x, params, g);
}

void gradientQ (Dataset *data, double *dQdAlpha, double *dQdBeta)
{
	int i, j;
	int idx;

	/* This comes from the priors */
	/*RandoPriors*/
	/*
	for (i = 0; i < data->numLabelers; i++) {
		dQdAlpha[i] = - 1 * (data->alpha[i] - data->priorAlpha[i]);
	}

	for (j = 0; j < data->numImages; j++) {
		dQdBeta[j] = - 5 * (data->beta[j] - data->priorBeta[j]);
	}
	*/
	/*Here's where the real shit starts*/
	for (idx = 0; idx < data->numLabels; idx++) {
		int i = data->labels[idx].labelerId;
		int j = data->labels[idx].imageIdx;
		int k = data->labels[idx].workflowId;
		int alphaidx = k * data->numLabelers + i;
		int betaidx = k * data->numImages + j;
		int lij = data->labels[idx].label;
		double sigma = prob(data->alpha[alphaidx], data->beta[betaidx]);

		double increa, increb;
		double weight = data->probZ1[j] * (lij - sigma) + data->probZ0[j] * (1 - lij - sigma);
		//double weight = data->probZ1[j] * (2 * lij - 1) + data->probZ0[j] * (-2 * lij + 1);
		increa = .5 * weight * log(1-data->beta[betaidx]) * pow( 1-data->beta[betaidx], 
									 data->alpha[alphaidx] );
		increb = -.5 * weight * data->alpha[alphaidx] * pow( 1-data->beta[betaidx], 
								     data->alpha[alphaidx]-1 );
		dQdAlpha[alphaidx] += increa;
		dQdBeta[betaidx] += increb;
		//printf("Alpha: %f Beta: %f", data->alpha[alphaidx], data->beta[betaidx]);
		//printf("Blah%f %f %f\n", log(1-data->beta[j]), 1-data->beta[j], data->alpha[i]);
		//printf("%d %d %d %f %f %f %f\n", i, j, lij, sigma, increa, increb, weight); 
		//printf("%f %f\n", data->probZ1[j], data->probZ0[j]);
		/*
		if ( isnan(prob(data->alpha[i], data->beta[j])) )
{
printf("%d %d\n", i, j);
printf("%f %f\n", data->probZ1[j], data->probZ0[j]);
printf("alpha %f beta %f\n", data->alpha[i], data->beta[j]);
printf("sigma %f\n", sigma);
printf("label %d\n", lij);
printf("prob: %f\n", prob);
printf("%f\n", pow( 1-data->beta[j], data->alpha[i]-1 ));
printf("%f\n", data->alpha[i]);
printf("%f %f\n", increa, increb);
}
fflush(0);


	for (i = 0; i < data->numLabelers; i++) {
		printf("i %d %f %f\n", i, data->alpha[i], dQdAlpha[i]);
	}

	for (j = 0; j < data->numImages; j++) {
		printf("j %d %f %f\n", j, data->beta[j], dQdBeta[j]);
	}
		*/
	  }
	
}

double computeLikelihood (Dataset *data)
{
	int i, j;
	int idx;
	double L = 0;

	double *alpha = data->alpha, *beta = data->beta;

	for (j = 0; j < data->numImages; j++) {
		double P1 = data->priorZ1[j];
		double P0 = 1 - data->priorZ1[j];
		for (idx = 0; idx < data->numLabels; idx++) {
			if (data->labels[idx].imageIdx == j) {
				int i = data->labels[idx].labelerId;
				int lij = data->labels[idx].label;
		double sigma = prob(alpha[i], beta[j]);
				P1 *= pow(sigma, lij) * pow(1 - sigma, 1 - lij);
				P0 *= pow(sigma, 1 - lij) * pow(1 - sigma, lij);
			}
		}
		L += log(P1 + P0);
	}

	/* Add Gaussian (standard normal) prior for alpha */
	for (i = 0; i < data->numLabelers; i++) {
		L += log(gsl_sf_erf_Z(alpha[i] - data->priorAlpha[i]));
	}

	/* Add Gaussian (standard normal) prior for beta */
	for (j = 0; j < data->numImages; j++) {
		L += log(gsl_sf_erf_Z(beta[j] - data->priorBeta[j]));
	}

	return L;
}

double computeQ (Dataset *data)
{
	int i, j;
	int idx;
	double Q = 0;

	double *alpha = data->alpha, *beta = data->beta;

	/* Start with the expectation of the sum of priors over all images */
	for (j = 0; j < data->numImages; j++) {
		Q += data->probZ1[j] * log(data->priorZ1[j]);
		Q += data->probZ0[j] * log(1 - data->priorZ1[j]);
	}

	for (idx = 0; idx < data->numLabels; idx++) {
		int i = data->labels[idx].labelerId;
		int j = data->labels[idx].imageIdx;
		int k = data->labels[idx].workflowId;
		int alphaidx = k * data->numLabelers + i;
		int betaidx = k * data->numImages + j;
		int lij = data->labels[idx].label;
		/* Do some analytic manipulation first for numerical stability! */
		double logSigma = log( prob(alpha[alphaidx], beta[betaidx]) );
		double logOneMinusSigma = log( 1-prob(alpha[alphaidx], beta[betaidx]) );
		if (beta[betaidx] > 1 || beta[betaidx] < 0 || alpha[alphaidx] < 0) {
		  Q += -10000000000000000;
		} else {
		  Q += data->probZ1[j] * (lij * logSigma + (1 - lij) * logOneMinusSigma) +
		    data->probZ0[j] * ((1 - lij) * logSigma + lij * logOneMinusSigma);

		}
		if (isnan(Q)) { 
		  printf("%d %d %d %f %f %f %f %f %f %f %f\n", i, j, lij, logSigma, 
			 logOneMinusSigma,
			 prob(alpha[i], beta[j]),
			 data->probZ1[j],
			 data->probZ0[j],
			 alpha[alphaidx],
			 beta[betaidx],
			 Q);
		  abort(); 
		}
	}

	/* Add Gaussian (standard normal) prior for alpha */
	for (i = 0; i < data->numLabelers * data->numWorkflows; i++) {
		//Q += log(gsl_sf_erf_Z(alpha[i] - data->priorAlpha[i]));
	}

	/* Add Gaussian (standard normal) prior for beta */
	for (j = 0; j < data->numImages * data->numWorkflows; j++) {
		//Q += log(gsl_sf_erf_Z(beta[j] - data->priorBeta[j]));
	}

	// add penalty of beta not between 0 and 1
	for (j = 0; j < data->numImages * data->numWorkflows; j++) {
	  if ( data->beta[j] > 1 || data->beta[j] < 0 ) {
	    Q -= 100;
	  }
	}
	return Q;
}

double logProbL (int l, int z, double alphaI, double betaJ)
{
	double p;

	if (z == l) {
		p = log(prob(alphaI, betaJ));
	} else {
		p = log(1-prob(alphaI, betaJ));
	}
/*
  printf("p: %f\n", prob(alphaI, betaJ) );
  printf("1-p: %f\n", 1-prob(alphaI, betaJ) );
  printf("log: %f\n", p);
*/
	return p;
}

double prob(double alpha, double beta)
{
  return .5 + .5 * pow(1-beta, alpha);
}

void EStep (Dataset *data)
{
	int j;
	int idx;
/*
  printf("begin Estep\n");
  fflush(0);
*/
	for (j = 0; j < data->numImages; j++) {
		data->probZ1[j] = log(data->priorZ1[j]);
		data->probZ0[j] = log(1 - data->priorZ1[j]);
		//printf("%d %f %f\n", j, data->probZ1[j], data->probZ0[j]);
	}

	for (idx = 0; idx < data->numLabels; idx++) {
		int i = data->labels[idx].labelerId;
		int j = data->labels[idx].imageIdx;
		int k = data->labels[idx].workflowId;
		int alphaidx = k * data->numLabelers + i;
		int betaidx = k * data->numImages + j;
		int lij = data->labels[idx].label;
		//printf("JOE%d %d %d %f %f\n", k, alphaidx, betaidx, data->alpha[alphaidx], data->beta[betaidx]); 
		data->probZ1[j] += logProbL(lij, 1, data->alpha[alphaidx], 
					    data->beta[betaidx]);
		data->probZ0[j] += logProbL(lij, 0, data->alpha[alphaidx], 
					    data->beta[betaidx]);
		//printf("%d %d %f %f\n", j, lij, data->probZ1[j], data->probZ0[j]);
	}

	/* Exponentiate and renormalize */
	for (j = 0; j < data->numImages; j++) {
	  //printf("Before: %f %f\n", data->probZ1[j], data->probZ0[j]);
	  /*Chris adding code here to do this better */
	  double normalizingDenominator = data->probZ1[j] + 
	    log(1 + exp(data->probZ0[j] -
			data->probZ1[j]));
	  data->probZ1[j] = data->probZ1[j] - normalizingDenominator;
	  data->probZ0[j] = data->probZ0[j] - normalizingDenominator;
	  //printf("Hmm: %f %f %f\n", normalizingDenominator, data->probZ1[j], data->probZ0[j]);
	  data->probZ1[j] = exp(data->probZ1[j]);
	  data->probZ0[j] = exp(data->probZ0[j]);
	  /*
	    data->probZ1[j] = exp(data->probZ1[j]);
	    data->probZ0[j] = exp(data->probZ0[j]);
	    printf("%f %f\n", data->probZ1[j], data->probZ0[j]);
	    data->probZ1[j] = data->probZ1[j] / (data->probZ1[j] + data->probZ0[j]); 
	    data->probZ0[j] = 1 - data->probZ1[j];
	  */
	  //printf("%d %f %f\n", j, data->probZ1[j], data->probZ0[j]);
	  if (isnan(data->probZ1[j])) {
	    printf("ABORTING?");
	    abort();
	  }
	}
/*
  printf("end Estep\n");
  fflush(0);
*/
}

void MStep (Dataset *data)
{
	double lastF;
	int iter, status;
	int i, j;

	int numLabelers = data->numLabelers;
	int numImages = data->numImages;
	int numWorkflows = data->numWorkflows;
		
	gsl_vector *x;
	gsl_multimin_fdfminimizer *s;
	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_function_fdf my_func;

	x = gsl_vector_alloc((numLabelers * numWorkflows) + (numImages * numWorkflows));

	/* Pack alpha and beta into a gsl_vector */
	packX(x, data);

	/* Initialize objective function */
	my_func.n = (numLabelers * numWorkflows) + (numImages * numWorkflows);
	my_func.f = &my_f;
	my_func.df = &my_df;
	my_func.fdf = &my_fdf;
	my_func.params = data;

	/* Initialize minimizer */
	T = gsl_multimin_fdfminimizer_conjugate_pr;
	s = gsl_multimin_fdfminimizer_alloc(T, (numLabelers * numWorkflows) + (numImages * numWorkflows));

	gsl_multimin_fdfminimizer_set(s, &my_func, x, 0.01, 0.01);
	iter = 0;
	do {
		lastF = s->f;
		iter++;
		/*printf("iter=%d f=%f\n", iter, s->f);*/
	
		status = gsl_multimin_fdfminimizer_iterate(s);
		if (status) {
			break;
		}
		status = gsl_multimin_test_gradient(s->gradient, 1e-3);
//	} while (lastF - s->f > 0.01 && status == GSL_CONTINUE && iter < 25);
	} while (iter < 5);

	/* Unpack alpha and beta from gsl_vector */
	unpackX(s->x, data);
	
	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(x);
}
