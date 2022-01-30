//#define KOKKOS_DONT_INCLUDE_CORE_CONFIG_H=1
//#include <Kokkos_Core.hpp>
//#include <Kokkos_Sort.hpp>
//#include <KokkosKernels_Utils.hpp>
#include <KokkosKernels_Sorting.hpp>
//#include <KokkosKernels_default_types.hpp>
//#include <KokkosSparse_CrsMatrix.hpp>
//#include <Kokkos_ArithTraits.hpp>
//#include <Kokkos_Complex.hpp>
#include <cstdlib>

extern "C" void kokkos_start (int narg, char *args[])
{
  Kokkos::initialize(narg,args);
}

extern "C" void kokkos_stop() {
  Kokkos::finalize();
}

template<typename ValView, typename OrdView>
struct TestTeamBitonicFunctor
{
  typedef typename ValView::value_type Value;

  TestTeamBitonicFunctor(ValView& values_, OrdView& counts_, OrdView& offsets_)
    : values(values_), counts(counts_), offsets(offsets_)
  {}

  template<typename TeamMem>
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem t) const
  {
    int i = t.league_rank();
    KokkosKernels::TeamBitonicSort<int, Value, TeamMem>(values.data() + offsets(i), counts(i), t);
  }

  ValView values;
  OrdView counts;
  OrdView offsets;
};

extern "C" void quickSortListKokkos(size_t arr[], size_t stack[], size_t num_arrays, size_t array_size)
{
  size_t i, j;

  size_t num_values = num_arrays * array_size;

  typedef Kokkos::View<size_t *> OrdView;
  typedef Kokkos::View<size_t *> ValView;

#ifdef DEBUG
  printf("quickSortListKokkos: Creating views\n");
#endif
  OrdView d_offsets("d_offsets",num_arrays);
  Kokkos::View<size_t *>::HostMirror h_offsets = Kokkos::create_mirror_view(d_offsets);
  OrdView d_counts("d_counts",num_arrays);
  Kokkos::View<size_t *>::HostMirror h_counts = Kokkos::create_mirror_view(d_counts);

  ValView d_values("d_values",num_values);
  //  Kokkos::View<size_t *>::HostMirror h_values = Kokkos::create_mirror_view(d_values);
#ifdef DEBUG
  printf("quickSortListKokkos: Done creating views\n");
#endif

#ifdef DEBUG
  printf("quickSortListKokkos: Initializing offset and counts arrays on host\n");
#endif

  for (i=0; i<num_arrays;i++) {
    h_offsets[i] = (int)(i*array_size);
    h_counts[i] = array_size;
  }
#ifdef DEBUG
  printf("quickSortListKokkos: Initializing data array on host\n");
#endif
  Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_values(arr,num_values);
  // for (i=0; i<num_values;i++) {
  //   h_values[i] = arr[i];
  // }

#ifdef DEBUG
  printf("quickSortListKokkos: Copying arrays to device\n");
#endif

  Kokkos::deep_copy(d_offsets,h_offsets);
  Kokkos::deep_copy(d_counts,h_counts);
  Kokkos::deep_copy(d_values,h_values);

#ifdef DEBUG
  printf("quickSortListKokkos: Performing sort\n");
#endif

  Kokkos::parallel_for(Kokkos::TeamPolicy<>(num_arrays, Kokkos::AUTO()),
      TestTeamBitonicFunctor<ValView, OrdView>(d_values, d_counts, d_offsets));
  Kokkos::deep_copy(h_values, d_values);

  // for (i=0; i<num_values;i++) {
  //   arr[i] = h_values[i];
  // }

#ifdef DEBUG
  printf("quickSortList: num_arrays, array_size = %ld, %ld\n",num_arrays,array_size);
#endif
}


