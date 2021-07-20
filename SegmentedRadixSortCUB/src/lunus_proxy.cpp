/* LUNUS_PROXY.C - Test code to optimize assimilation filter GPU code
   
   Author: Mike Wall 
   Date: 5/2021
   Version: 1.
   
   "lunus_proxy <image in> <image out>""

   */

#include "lunus_proxy.h"

int main(int argc, char *argv[])
{
  printf("\nEnters lunus_proxy.\n\n");
  FILE
	*imagein,
	*imageout;
  
  DIFFIMAGE 
	*imdiff;

  int
    mask_size,
    binsize;
/*
 * Set input line defaults:
 */
	
  mask_size = 15;  // 3 -Pierre
  binsize = 1;

/*
 * Read information from input line:
 */
	switch(argc) {
	  case 3:
	  if (strcmp(argv[2], "-") == 0) {
	    imageout = stdout;
	  }
	  else {
	    if ( (imageout = fopen(argv[2],"wb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[2]);
	      exit(0);
	    }
	  }
	  case 2:
	  if (strcmp(argv[1], "-") == 0) {
	    imagein = stdin;
	  }
	  else {
	    if ( (imagein = fopen(argv[1],"rb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[1]);
	      exit(0);
	    }
	  }
	  break;
	  default:
	  printf("\n Usage: modeim "
		 "<image in> <image out>\n\n");
	  exit(0);
	}
  
	imdiff = (DIFFIMAGE *)malloc(sizeof(DIFFIMAGE));
	imdiff->num_panels=1;
	imdiff->this_panel=0;
	imdiff->ignore_tag=0x7fff;
	imdiff->overload_tag=0x7fff;
	imdiff->mask_tag=0x7fff;
	imdiff->error_msg = (char *)malloc(sizeof(char)*1024);
	imdiff->hpixels = 1024;
	imdiff->vpixels = 1024;
	imdiff->big_endian = 0;

/*
 * Read diffraction image:
 */

  imdiff->infile = imagein;
  if (lreadim(imdiff) != 0) {
    perror(imdiff->error_msg);
    goto CloseShop;
  }

  /*
   * Initialize mask parameters:
   */

  imdiff->mode_height = mask_size - 1;
  imdiff->mode_width = mask_size - 1;
  imdiff->mode_binsize = binsize;

/*
 * Mode image:
 */

  lmodeim(imdiff);

/*
 * Write the output image:
 */

  imdiff->outfile = imageout;
  if(lwriteim(imdiff) != 0) {
    perror(imdiff->error_msg);
    goto CloseShop;
  }

CloseShop:
  free(imdiff->error_msg);
  free(imdiff->image);
  free(imdiff);

/*
 * Close files:
 */
  
  fclose(imagein);
  fclose(imageout);
  
}

