default: em

clean:
	rm -f *.o em

# Set these variables to the point to the correct locations

#CBLAS_LIB=~/usr/local/lib
#GSL_LIB=~opt/local/include/
GSL_INCLUDE=/opt/local/include/

#MATLAB=/projects/matlab7

#em_mex: em_mex.c em.c data.c data.h prob_functions.c prob_functions.h
#	$(MATLAB)/bin/mex $(GSL_LIB)/libgsl.a $(CBLAS_LIB)/libgslcblas.a -I$(GSL_INCLUDE) -arch=maci64 em_mex.c em.c data.c prob_functions.c
#
#em: em.c data.c data.h prob_functions.c prob_functions.h
#	gcc -O3 $(GSL_LIB)/libgsl.a $(CBLAS_LIB)/libgslcblas.a -I$(GSL_INCLUDE) #-std=c99 -o em em.c data.c prob_functions.c -lm
 #-m64
 
#em_mex: em_mex.c em.c data.c data.h prob_functions.c prob_functions.h  $(MATLAB)/extern/include/mex.h
#	 $(MATLAB)/bin/mex -I $(GSL_INCLUDE) em_mex.c em.c data.c prob_functions.c -lgsl -lgslcblas -lm

em: em.c data.c data.h prob_functions.c prob_functions.h verif.h
	gcc -O3 -I $(GSL_INCLUDE) -L/opt/local/lib -std=c99 -o em0 em.c data.c prob_functions.c verif.c -lm -lgsl -lgslcblas
