#include <stdio.h>
#include <stdlib.h>
#include "reader.h"

int main() {
  //const float* v = json_bonsai_reader("/data1/LEAPSData/LEAPS1b/bonsai_simulations/s_0.5_m_0.5_lMW_0_bMW_90/params.json");
  json_bonsai_reader("/data1/LEAPSData/LEAPS1b/bonsai_simulations/s_0.5_m_0.5_lMW_0_bMW_90/params.json");
  printf("\n");
  /*int i;
  for(i=0;i<12;++i) {
    printf("%f\n",v[i]);
  }
  free((float*)v);
  */
  return 0;
}
