/*for compilation, link the following library: jsmn/libjsmn.a

jsmn library structures (documentation: https://github.com/zserge/jsmn):

typedef struct {
  jsmntype_t type; // Token type
  int start;       // Token start position
  int end;         // Token end position
  int size;        // Number of child (nested) tokens
} jsmntok_t;

typedef enum {
JSMN_UNDEFINED = 0,
JSMN_OBJECT = 1,
JSMN_ARRAY = 2,
JSMN_STRING = 3,
JSMN_PRIMITIVE = 4
} jsmntype_t;
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "jsmn/jsmn.h"

#define BUFFER_SIZE 500
#define INIT_TOKEN 4
#define FINAL_TOKEN 26

void read_file(char* filepath, char* fileContent) {
  int c;
  int index=0;
  FILE* f = fopen(filepath, "rt");  
  while((c = fgetc(f)) != EOF){
    fileContent[index] = c;
    index++;
  }
  fileContent[index] = '\0';  
}

int get_json_bonsai_reader_size() {
  return (FINAL_TOKEN-INIT_TOKEN)/2 + 1;
}

/*returns an array with all parameters in the json files of the bonsai simulations:
0 vr
1 vt
2 vt_phi
3 size_ratio
4 mass_ratio
5 Rsep
6 lMW
7 bMW
8 lM31
9 bM31
10 lR
11 bR
Note: the pointer has to be freed by the caller! Do 'free((float*)pointer)'
*/
float* json_bonsai_reader(char* file, float *out) {
  char json_str[BUFFER_SIZE];
  read_file(file, json_str);

  jsmn_parser parser;
  jsmntok_t tokens[100]; //this can be optimized
  jsmn_init(&parser);
  jsmn_parse(&parser, json_str, strlen(json_str), tokens, 50);

  int i_tok;
  for(i_tok=INIT_TOKEN; i_tok<=FINAL_TOKEN; i_tok=i_tok+2) {
    jsmntok_t key = tokens[i_tok];
    unsigned int l = key.end - key.start;   
    char* key_str = (char*)malloc((l+1)*sizeof(char));
    memcpy(key_str, &json_str[key.start], l);
    key_str[l] = '\0'; 
    int out_idx = (i_tok-INIT_TOKEN)/2;
    out[out_idx]=(float)atof(key_str);
    free((char*)key_str);
  }
  return out;
}
