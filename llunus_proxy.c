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

double ltime() {
  double t;
#ifdef USE_OPENMP
  t = omp_get_wtime();
  //  t = 0.0;
#else
  t = ((double)clock())/CLOCKS_PER_SEC;
#endif
  return t;
}

#ifdef USE_OPENMP
#include<omp.h>
#endif
#include<time.h>

#ifdef USE_OFFLOAD
#pragma omp declare target
#endif

double log_local(double x) {
  int n,N=10,onefac=1;
  double y,xfac,f = 0.0;

  xfac = y = x-1.;

  for (n=1;n<=10;n++) {
    f += (double)onefac*xfac/(double)n;
    onefac *= -1;
    xfac *= y;
  }
  return(f);
}

void swap(size_t *a, size_t *b) 
{ 
  size_t t = *a; 
  *a = *b; 
  *b = t; 
} 
  
int partition(size_t arr[], int l, int h) 
{ 
  size_t x = arr[h]; 
  int i = (l - 1); 
  int j;

  for (j = l; j <= h - 1; j++) { 
    if (arr[j] <= x) { 
      i++; 
      swap(&arr[i], &arr[j]); 
    } 
  } 
  swap(&arr[i + 1], &arr[h]); 
  return (i + 1); 
} 

void quickSortIterative(size_t arr[], size_t stack[], int l, int h) 
{ 
  // initialize top of stack 
  int top = -1; 
  
  // push initial values of l and h to stack 
  stack[++top] = l; 
  stack[++top] = h; 
  
  // Keep popping from stack while is not empty 
  while (top >= 0) { 
    // Pop h and l 
    h = stack[top--]; 
    l = stack[top--]; 
  
    // Set pivot element at its correct position 
    // in sorted array 
    int p;
    p = partition(arr, l, h); 
  
    // If there are elements on left side of pivot, 
    // then push left side to stack 
    if (p - 1 > l) { 
      stack[++top] = l; 
      stack[++top] = p - 1; 
    } 
  
    // If there are elements on right side of pivot, 
    // then push right side to stack 
    if (p + 1 < h) { 
      stack[++top] = p + 1; 
      stack[++top] = h; 
    } 
  } 
}

void insertion_sort(size_t *a,int first,int last) {
  int len;
  int i=1+first;
  len = last - first + 1;
  while (i < len) {
    size_t x = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > x) {
      a[j + 1] = a[j];
      j = j - 1;
    }
    a[j+1] = x;
    i = i + 1;
  }
}

void quicksort(size_t *a,int first,int last){
  int i, j, pivot;
  size_t temp;

  if(first<last){
    pivot=first;
    i=first;
    j=last;

    while(i<j){
      while(a[i]<=a[pivot]&&i<last)
	i++;
      while(a[j]>a[pivot])
	j--;
      if(i<j){
	temp=a[i];
	a[i]=a[j];
	a[j]=temp;
      }
    }

    temp=a[pivot];
    a[pivot]=a[j];
    a[j]=temp;
    quicksort(a,first,j-1);
    quicksort(a,j+1,last);

  }
}
#ifdef USE_OFFLOAD
#pragma omp end declare target
#endif

int lmodeim(DIFFIMAGE *imdiff_in)
{
  RCCOORDS_DATA 
    half_height,
    half_width,
    n,
    m; 

  static IMAGE_DATA_TYPE
    //IMAGE_DATA_TYPE
    *image = NULL;

  IMAGE_DATA_TYPE
    maxval,
    minval,
    binsize;
  
  int
    return_value = 0;

  static size_t
    //size_t
    *image_mode = NULL,
    *window = NULL,
    *stack = NULL;

  DIFFIMAGE *imdiff;

  size_t wlen = (imdiff_in->mode_height+2)*(imdiff_in->mode_width+2);
    
  int pidx;

  size_t
    i,
    j;

  static size_t 
    //size_t 
    image_length = 0,
    hpixels = 0,
    vpixels = 0;

  size_t num_teams = LUNUS_TEAMS;
  size_t num_threads = LUNUS_THREADS;

  //  printf("omp_get_num_teams() = %d, omp_get_num_threads() = %d\n",omp_get_num_teams(), omp_get_num_threads());

  //  fflush(stdout);

  imdiff = &imdiff_in[0];

  half_height = imdiff->mode_height / 2;
  half_width = imdiff->mode_width / 2;
  image_length = imdiff->image_length;
  hpixels = imdiff->hpixels;
  vpixels = imdiff->vpixels;

    //    printf("MODEIM: Allocating arrays\n");

    image = (IMAGE_DATA_TYPE *)calloc(image_length,sizeof(IMAGE_DATA_TYPE));
    image_mode = (size_t *)calloc(image_length,sizeof(size_t));
    window = (size_t *)calloc(wlen*num_teams*num_threads,sizeof(size_t));
    stack = (size_t *)calloc(wlen*num_teams*num_threads,sizeof(size_t));

    /* 
     * Allocate working mode filetered image: 
     */ 
  
    if (!image || !image_mode || !window || !stack) {
      sprintf(imdiff->error_msg,"\nLMODEIM:  Couldn't allocate arrays.\n\n");
      return_value = 1;
      goto CloseShop;
    }

#ifdef USE_OFFLOAD
#pragma omp target enter data map(alloc:image[0:image_length],image_mode[0:image_length])
#pragma omp target enter data map(alloc:window[0:wlen*num_teams*num_threads],stack[0:wlen*num_teams*num_threads])
#endif


  //  printf("LMODEIM: mode_height = %d, mode_width = %d, num_panels = %d\n",imdiff_in->mode_height,imdiff_in->mode_width,imdiff_in->num_panels);

  for (pidx = 0; pidx < imdiff_in->num_panels; pidx++) {
    size_t num_mode_values=0, num_median_values=0, num_med90_values=0, num_this_values=0;
    imdiff = &imdiff_in[pidx];
    if (pidx != imdiff->this_panel) {
      perror("LMODEIM: Image panels are not indexed sequentially. Aborting\n");
      exit(1);
    }
    if (hpixels != imdiff->hpixels || vpixels != imdiff->vpixels) {
      perror("LMODEIM: Image panels are not identically formatted. Aborting\n");
      exit(1);
    }

    memcpy(image, imdiff->image, image_length*sizeof(IMAGE_DATA_TYPE));

    IMAGE_DATA_TYPE overload_tag = imdiff->overload_tag;
    IMAGE_DATA_TYPE ignore_tag = imdiff->ignore_tag;
    

    // Compute min and max for image

    int got_first_val = 0;
    
    size_t index; 

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

    // Compute the mode filtered image


    size_t tm, th, j, i,tidx;

    size_t ntidx = num_teams*num_threads;

#ifdef USE_OFFLOAD
#pragma omp target update to(image[0:image_length],image_mode[0:image_length])
    //#pragma omp target update to(window[0:wlen*num_teams*num_threads],stack[0:wlen*num_teams*num_threads])
#endif

    //    printf("Entering omp offload region\n");
    //    fflush(stdout);
    
    double start = ltime();
    //double start =  0.0;

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target map(to:minval,binsize,wlen,overload_tag,ignore_tag,num_teams,num_threads) 
#pragma omp teams
#pragma omp distribute parallel for collapse(2) schedule(static,1)
#else
    //#pragma omp distribute parallel for collapse(2)
#pragma omp parallel for shared(stack,window,image,image_mode) private(i,j,th)
#endif
#endif
    for (tm = 0; tm < num_teams; tm++) {
      for (th = 0; th < num_threads; th++) {
	size_t jlo = half_height + tm;
	size_t jhi = vpixels - half_height;     
	size_t ilo = half_width + th;
	size_t ihi = hpixels - half_width;     
      for (j = jlo; j < jhi; j=j+num_teams) {
	for (i = ilo; i < ihi; i=i+num_threads) {
	    int mode_ct = 0;
	    size_t mode_value=0, max_count=0;
	    size_t index_mode = j*hpixels + i;
	    size_t nt = num_threads;
	    size_t ntm = num_teams;
	    size_t this_value = (image[index_mode]-minval)/binsize + 1;
	    if (ntm != num_teams || nt != num_threads) {
	      printf("Number of teams, threads %ld, %ld differs from assumed numbers %ld, %ld\n",ntm,nt,num_teams,num_threads);
	      exit(1);
	    }
	    size_t *this_window = &window[(tm*nt+th)*wlen];
	    size_t *this_stack = &stack[(tm*nt+th)*wlen];
	    size_t k;
	    for (k = 0; k < wlen; k++) {
	      this_window[k] = 0;
	    }
	    int l = 0;
	    //        printf("Start tm = %ld,th = %ld,i = %d,j = %ld\n",tm,th,i,index_mode/hpixels);
	    RCCOORDS_DATA r, c, rlo, rhi, clo, chi;
	    rlo = j - half_height;
	    rhi = (j + half_height);
	    clo = i - half_width;
	    chi = (i + half_width);
	    for (r = rlo; r <= rhi; r++) {
	      for (c = clo; c <= chi; c++) {
		size_t index = r*hpixels + c;
		if ((image[index] != overload_tag) &&
		    (image[index] != ignore_tag) &&
		    (image[index] < MAX_IMAGE_DATA_VALUE)) {
		  this_window[l] = (image[index]-minval)/binsize + 1;
		  l++;
		}
	      }
	    }
	    if (l == 0 || image[index_mode] == ignore_tag || image[index_mode] == overload_tag || image[index_mode] >= MAX_IMAGE_DATA_VALUE) {
	      image_mode[index_mode] = 0;
	    }
	    else {
	      //          printf("Starting quicksort for i=%d,j=%ld\n",i,index_mode/hpixels);
	      //	  insertion_sort(this_window,0,l-1);
	      //	  quicksort(this_window,0,l-1);
	      quickSortIterative(this_window,this_stack,0,l-1);
	      //          printf("Done with quicksort for i=%d,j=%ld\n",i,index_mode/hpixels);
	      // Get the median
	      int kmed = l/2;
	      int k90 = l*9/10;
	      size_t min_value = this_window[0];
	      size_t max_value = this_window[l-1];
	      size_t range_value = max_value - min_value;
	      size_t median_value = this_window[kmed];
	      size_t med90_value = this_window[k90];

	      // Get the mode
	      size_t this_count = 1;
	      size_t last_value = this_window[0];
	      max_count = 1;
	      size_t k;
	      for (k = 1; k < l; k++) {
		if (this_window[k] == last_value) {
		  this_count++;
		} else {
		  last_value = this_window[k];
		  this_count = 1;
		}
		if (this_count > max_count) max_count = this_count;
	      }
	      this_count = 1;
	      last_value = this_window[0];
	      double entropy = 0.0;
	      for (k = 1; k < l; k++) {
		if (this_window[k] == last_value) {
		  this_count++;
		} else {
		  double p = (double)this_count/(double)l;
		  entropy -=  p * log_local(p);
		  last_value = this_window[k];
		  this_count = 1;
		}
		if (this_count == max_count) {
		  mode_value += this_window[k];
		  mode_ct++;
		}
	      }
	      mode_value = (size_t)(((float)mode_value/(float)mode_ct) + .5);
	      double p = (double)this_count/(double)l;
	      entropy -=  p * log_local(p);
	      //	  image_mode[index_mode] = (size_t)(((float)mode_value/(float)mode_ct) + .5);
#ifdef DEBUG
	      if (j == 2200 && i == 1800) {
		printf("LMODEIM: entropy = %g, mode_ct = %d, mode_value = %ld, median_value = %ld, range_value = %ld, this_value = %ld, med90_value = %ld, kmed = %d, k90 = %d\n",entropy,mode_ct,mode_value,median_value,range_value,this_value,med90_value,kmed,k90);
	      } 
#endif 
	      // Depending on the distribution, use alternative values other than the mode
		if (range_value <= 2) {
		  image_mode[index_mode]  = this_value;
		  num_this_values++;
		} else {
		  if (this_value <= med90_value) {
		    image_mode[index_mode] = this_value;
		    num_this_values++;
		  } else {
		    image_mode[index_mode] = this_window[(size_t)(((double)k90*(double)rand())/(double)RAND_MAX)];
		    num_med90_values++;
		  //	      printf("%d %ld %ld\n",kmed,mode_value,median_value);
		  //	      mode_value = median_value;
		  }
		}
		/*	      if (entropy > log(10.)) {
		if (mode_ct == 1) {
		  image_mode[index_mode] = median_value;
		  num_median_values++;
		} else {
		  image_mode[index_mode] = mode_value;
		  num_mode_values++;
		}
	      } else {
		if (range_value <= 2) {
		  image_mode[index_mode]  = this_value;
		  num_this_values++;
		} else {
		  if (this_value <= med90_value) {
		    image_mode[index_mode] = this_value;
		    num_this_values++;
		  } else {
		    image_mode[index_mode] = this_window[(size_t)(((double)k90*(double)rand())/(double)RAND_MAX)];
		    num_med90_values++;
		  //	      printf("%d %ld %ld\n",kmed,mode_value,median_value);
		  //	      mode_value = median_value;
		  }
		}
	      }
		*/
	    }
	    //        printf("Stop tm = %ld,th = %ld,i = %d,j = %d\n",tm,th,i,j);
	  }
	}
      }
    }

    // Now image_mode holds the mode filtered values
    // Convert these values to pixel values and store them in the input image

    double stop = ltime();

    double tel = stop-start;
    fflush(stdout);

#ifdef DEBUG
    printf("LMODEIM: %g seconds, num_mode_values=%ld,num_median_values=%ld,num_this_values=%ld,num_med90_values=%ld\n",tel,num_mode_values,num_median_values,num_this_values,num_med90_values);
#endif
    
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target teams distribute map(to:minval,binsize,ignore_tag)
#else
#pragma omp distribute
#endif
#endif

    for (j = 0; j < vpixels; j++) {
      if (j < half_height || j > (vpixels-half_height)) {
	for (i = 0; i < hpixels; i++) {
	  size_t this_index = j * hpixels + i;
	  image[this_index] = ignore_tag;
	} 
      } else {

	for (i = 0; i < half_width; i++) {
	  size_t this_index = j * hpixels + i;
	  image[this_index] = ignore_tag;
	}
	
	for (i = hpixels - half_width; i < hpixels; i++) {
	  size_t this_index = j * hpixels + i;
	  image[this_index] = ignore_tag;
	}

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp parallel for schedule(static,1)
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
    }
	  
#ifdef USE_OFFLOAD
#pragma omp target update from(image[0:image_length]) 
#endif
    memcpy(imdiff->image,image,image_length*sizeof(IMAGE_DATA_TYPE));

#ifdef USE_OFFLOAD
#pragma omp target exit data map(delete:image[0:image_length],image_mode[0:image_length],window[0:wlen*num_threads*num_teams],stack[0:wlen*num_threads*num_teams])
#endif
      //      printf("MODEIM: Freeing arrays\n");
      free(image);
      free(image_mode);
      free(window);
      free(stack);
  }
 CloseShop:
  return(return_value);
}

int lmodeim_old(DIFFIMAGE *imdiff_in)
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

