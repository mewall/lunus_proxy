#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <string.h>
#include<stdint.h>
#include<math.h>
#include<errno.h>
#include<limits.h>
#ifdef USE_OPENMP
#include<omp.h>
#endif

#define MAX_IMAGE_DATA_VALUE 32767	/* Maximum value of pixel in image */
#define SMV_IGNORE_TAG 32767
#define DEFAULT_HEADER_LENGTH 512
#define SMV_MAX SHRT_MAX
#define SMV_MIN SHRT_MIN

#ifndef LUNUS_TEAMS
#define LUNUS_TEAMS 64
#endif
#ifndef LUNUS_THREADS
#define LUNUS_THREADS 32
#endif

typedef short RCCOORDS_DATA;
typedef short IMAGE_DATA_TYPE;
typedef short SMV_DATA_TYPE;

typedef struct 
{
  FILE *infile;
  FILE *outfile;
  char big_endian;
  int num_panels;               /* number of panels in image */
  int this_panel;               /* index of this panel */
  char *header;		        /* Image header */
  size_t header_length;	        /* Length of image header (4096 default) */
  IMAGE_DATA_TYPE *image;	/* Pointer to image */
  size_t image_length;	        /* Total number of pixels in image */
  short vpixels;		/* Number of vertical pixels */
  short hpixels;		/* Number of horizontal pixels */
  IMAGE_DATA_TYPE ignore_tag;   /* Ignore this pixel value */
  IMAGE_DATA_TYPE overload_tag; /* Pixel value indicating ovld */
  IMAGE_DATA_TYPE mask_tag;     /* Value which mask puts in image */
  size_t mode_height;           /* Height of mode matrix */
  size_t mode_width;            /* Width of mode matrix */
  IMAGE_DATA_TYPE mode_binsize; /* Pixel value bin size for mode */
				/* filter */
  char *error_msg;
} DIFFIMAGE;

int lreadim(DIFFIMAGE *imdiff);
int lwriteim(DIFFIMAGE *imdiff);
int lmodeim(DIFFIMAGE *imdiff_in);
