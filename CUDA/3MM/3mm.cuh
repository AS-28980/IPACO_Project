/**
 * 3mm.cuh: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef THREEMM_H
# define THREEMM_H

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(N)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#  endif

#  ifdef SMALL_DATASET
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#  endif

#  ifdef LARGE_DATASET
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#  endif

#  ifdef EXTRALARGE_DATASET
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#  endif
# endif /* !N */

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)
# define _PB_NL POLYBENCH_LOOP_BOUND(NL,nl)
# define _PB_NM POLYBENCH_LOOP_BOUND(NM,nm)

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 16
#define DIM_THREAD_BLOCK_Y 16


#endif /* !THREEMM*/
