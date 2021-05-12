#include<lunus_proxy.h>

int lreadim(DIFFIMAGE *imdiff)
{
	size_t
	  i,
		num_read;

	int
		return_value = 0;  

	char
	  *buf;

	buf = (char *)malloc(DEFAULT_HEADER_LENGTH); // TV6 TIFF header size is generous enough

  /*
   * Read diffraction image header
   */


  num_read = fread(buf, sizeof(char), DEFAULT_HEADER_LENGTH,
                imdiff->infile);
  if (num_read != DEFAULT_HEADER_LENGTH) {
	sprintf(imdiff->error_msg,"\nCouldn't read all of header.\n\n");
	return(1);
  }

  imdiff->header_length = DEFAULT_HEADER_LENGTH;
  imdiff->header = (char *)calloc(imdiff->header_length,sizeof(char));
  fseek(imdiff->infile,0,SEEK_SET);
  num_read = fread(imdiff->header, sizeof(char), imdiff->header_length,
		   imdiff->infile);
  if (num_read != imdiff->header_length) {
    sprintf(imdiff->error_msg,"\nCouldn't read all of header.\n\n");
    return(1);
  }
  imdiff->image_length = imdiff->hpixels*imdiff->vpixels;
  imdiff->image = (IMAGE_DATA_TYPE *)calloc(imdiff->image_length,sizeof(IMAGE_DATA_TYPE));
      
  /*
   * Read image:
   */
  SMV_DATA_TYPE *imbuf;
  imbuf = (SMV_DATA_TYPE *)malloc(imdiff->image_length*sizeof(SMV_DATA_TYPE));
  num_read = fread(imbuf, sizeof(SMV_DATA_TYPE),
		   imdiff->image_length, imdiff->infile);
  if (num_read != imdiff->image_length) {
    sprintf(imdiff->error_msg,"\nCouldn't read all of image.\n\n");
    return(3);
  }
  if (ferror(imdiff->infile) != 0) {
    sprintf(imdiff->error_msg,"\nError while reading image\n\n");
    return(4);
  }
  // Copy image to imdiff, with possible change in type
  for (i=0;i<imdiff->image_length;i++) {
    imdiff->image[i]=(IMAGE_DATA_TYPE)imbuf[i];
  }
  free(imbuf);

  return(0);
}

int lwriteim(DIFFIMAGE *imdiff)
{
  size_t
    i,
    num_wrote;

  int
    return_value = 0;  


  /*
   * Write image header
   */



  num_wrote = fwrite(imdiff->header, sizeof(char), imdiff->header_length,
		     imdiff->outfile);
  if (num_wrote != imdiff->header_length) {
    return_value = 1;
    sprintf(imdiff->error_msg,"\nCouldn't write all of header.\n\n");
  }
  if (ferror(imdiff->outfile) != 0) {
    return_value = 2;
    sprintf(imdiff->error_msg,"\nError while writing header.\n\n");
  }
    
  /*
   * Write SMV image:
   */
  //    int ct=0;
  SMV_DATA_TYPE *imbuf;
  imbuf = (SMV_DATA_TYPE *)malloc(imdiff->image_length*sizeof(SMV_DATA_TYPE));
  for (i = 0;i < imdiff->image_length; i++) {
    if (imdiff->image[i] < SMV_MIN || imdiff->image[i] > SMV_MAX) {
      imbuf[i] = SMV_IGNORE_TAG;
      //	ct++;
    } else {
      imbuf[i] = (SMV_DATA_TYPE)imdiff->image[i];
    }
  }
  //    printf("ct = %d",ct);
  num_wrote = fwrite(imbuf, sizeof(SMV_DATA_TYPE), 
		     imdiff->image_length, imdiff->outfile);

  if (num_wrote != imdiff->image_length) {
    return_value = 3;
    sprintf(imdiff->error_msg,"\nCouldn't write all of image.\n\n");
  }
  if (ferror(imdiff->outfile) != 0) {
    return_value = 4;
    sprintf(imdiff->error_msg,"\nError while writing image\n\n");
  }
  return(return_value);
}

int lmodeim(DIFFIMAGE *imdiff_in)
{
  RCCOORDS_DATA 
    half_height,
    half_width,
    n,
    m,
    r, 
    c; 

  IMAGE_DATA_TYPE
    *image,
    maxval,
    minval,
    binsize;
  
  int
    return_value = 0;

  size_t
    *image_mode,
    num_bins;

  DIFFIMAGE *imdiff;

  int pidx;

  size_t
    index,
    i,
    j,
    k,
    *distn,
    *window;

  for (pidx = 0; pidx < imdiff_in->num_panels; pidx++) {
    imdiff = &imdiff_in[pidx];
    if (pidx != imdiff->this_panel) {
      perror("LMODEIM: Image panels are not indexed sequentially. Aborting\n");
      exit(1);
    }

    image = imdiff->image;
    IMAGE_DATA_TYPE overload_tag = imdiff->overload_tag;
    IMAGE_DATA_TYPE ignore_tag = imdiff->ignore_tag;
    size_t image_length = imdiff->image_length;
    
    /* 
     * Allocate working mode filetered image: 
     */ 
  
    image_mode = (size_t *)calloc(imdiff->image_length,
				  sizeof(size_t));
    if (!image_mode) {
      sprintf(imdiff->error_msg,"\nLMODEIM:  Couldn't allocate arrays.\n\n");
      return_value = 1;
      goto CloseShop;
    }

    // Compute min and max for image

    int got_first_val = 0;
    
    
    for (index = 0; index < image_length; index++) {
      if ((image[index] != overload_tag) &&
	  (image[index] != ignore_tag) &&
	  (image[index] < MAX_IMAGE_DATA_VALUE)) {
	if (got_first_val != 0) {
	  if (image[index] < minval) minval = image[index];
	  if (image[index] > maxval) maxval = image[index];
	} else {
	  minval = image[index];
	  maxval = image[index];
	  got_first_val = 1;
	}
      }
    }
    // Allocate the distribution

    binsize = imdiff->mode_binsize;
    num_bins = (size_t)((maxval - minval)/binsize) + 2;

    // Compute the mode filtered image

    half_height = imdiff->mode_height / 2;
    half_width = imdiff->mode_width / 2;
    int hpixels = imdiff->hpixels;
    int vpixels = imdiff->vpixels;

    size_t wlen = (imdiff->mode_height+1)*(imdiff->mode_width+1);
    

    size_t num_teams;
    size_t num_threads;

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target teams distribute map(from:num_teams,num_threads)
#else
#pragma omp distribute
#endif
#endif

    for (j = half_height; j < vpixels - half_height; j++) {
      num_teams = omp_get_num_teams();

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp parallel num_threads(32)
#else
#pragma omp parallel
#endif
#endif

      {
	num_threads = omp_get_num_threads();
      }
    }
    
    printf(" Number of teams, threads = %ld, %ld\n",num_teams,num_threads);

    window = (size_t *)calloc(wlen*num_teams*num_threads,sizeof(size_t));
    distn = (size_t *)calloc(num_bins*num_teams*num_threads,sizeof(size_t));
    
#ifdef USE_OFFLOAD
#pragma omp target enter data map(alloc:image[0:image_length],image_mode[0:image_length])
#pragma omp target update to(image[0:image_length],image_mode[0:image_length])
#pragma omp target enter data map(alloc:window[0:wlen*num_teams*num_threads],distn[0:num_bins*num_teams*num_threads])
#pragma omp target update to(window[0:wlen*num_teams*num_threads],distn[0:num_bins*num_teams*num_threads])
#endif
    
    clock_t start = clock();

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target teams distribute map(to:minval,binsize,num_bins,wlen,overload_tag,ignore_tag)
#else
#pragma omp distribute
#endif
#endif

    for (j = half_height; j < vpixels-half_height; j++) {

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp parallel for private(index,k,r,c) schedule(static,1) num_threads(32)
#else
#pragma omp parallel for private(index,k,r,c)
#endif
#endif

      for (i = half_width; i < hpixels-half_width; i++) {
	int mode_ct = 0;
	size_t mode_value=0, max_count=0;
	size_t index_mode = j*hpixels + i;
	size_t tm = omp_get_team_num();
	size_t th = omp_get_thread_num();
	size_t nt = omp_get_num_threads();
	size_t *this_distn = &distn[(tm*nt+th)*num_bins];
	size_t *this_window = &window[(tm*nt+th)*wlen];
	int l = 0;
	for (r = j - half_height; r <= j + half_height; r++) {
	  for (c = i - half_width; c <= i + half_width; c++) {
	    index = r*hpixels + c;
	    if ((image[index] != overload_tag) &&
		(image[index] != ignore_tag) &&
		(image[index] < MAX_IMAGE_DATA_VALUE)) {
	      this_window[l] = (image[index]-minval)/binsize + 1;
	      this_distn[this_window[l]]++;
	      l++;
	    }
	  }
	}
	if (l == 0) {
	  image_mode[index_mode] = 0;
	}
	else {
	  for (k = 0; k < l; k++) {
	    if (this_distn[this_window[k]] > max_count) {
	      max_count = this_distn[this_window[k]];
	    }
	  }
	  for (k = 0; k < l; k++) {
	    if (this_distn[this_window[k]] == max_count) {
	      mode_value += this_window[k];
	      mode_ct++;
	    }
	  }
	  image_mode[index_mode] = (size_t)(((float)mode_value/(float)mode_ct) + .5);
	  for (k = 0; k < l; k++) {
	    this_distn[this_window[k]] = 0;
	    this_window[k] = 0;
	  }
	}
      }
    }

    // Now image_mode holds the mode filtered values
    // Convert these values to pixel values and store them in the input image

    clock_t stop = clock();
    double tel = ((double)(stop-start))/CLOCKS_PER_SEC;

#ifdef DEBUG
    printf("kernel loop took %g seconds\n",tel);
#endif
    
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target teams distribute map(to:minval,binsize,ignore_tag)
#else
#pragma omp distribute
#endif
#endif

    for (j = half_height; j < vpixels - half_height; j++) {

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp parallel for schedule(static,1) num_threads(32)
#else
#pragma omp parallel for
#endif
#endif

      for (i = half_width; i < hpixels - half_width; i++) {
	size_t this_index = j * hpixels + i;
	if (image_mode[this_index] != 0) {
	  image[this_index] = (image_mode[this_index]-1)*binsize + minval;
	} else {
	  image[this_index] = ignore_tag;
	}
      }
    }

#ifdef USE_OFFLOAD
#pragma omp target exit data map(from:image[0:image_length]) map(delete:image_mode[0:image_length],window[0:wlen*num_threads*num_teams],distn[0:num_bins*num_teams*num_threads])
#endif

    free(image_mode);
    free(distn);
    free(window);
  }
 CloseShop:
  return(return_value);
}

