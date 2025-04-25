#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARAM_SIZE (1024 * 16)
#define TRAIN_SIZE (1024 * 16)
#define IO_SIZE (1024 * 4)
#define BUF_SIZE (1024 * 4)

typedef enum {
    RESULT_OK,
    RESULT_ERR,
} result_t;

typedef struct {
    int8_t param[PARAM_SIZE];
    int8_t buf1[BUF_SIZE];
    int8_t buf2[BUF_SIZE];
    char train[TRAIN_SIZE];
    char io[IO_SIZE];
} env_t;

env_t e;

result_t readfile(char* dst, const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open source file: %s\n", path);
        return RESULT_ERR;
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    fread(dst, 1, size, fp);
    fclose(fp);
    return RESULT_OK;
}

result_t train() {
    
}

int main() {
    if(readfile(e.train, "./train.txt") != RESULT_OK) {
        return 1;
    }
    if(readfile(e.io, "./input.txt") != RESULT_OK) {
        return 1;
    }
    if(train() != RESULT_OK) {
        return 1;
    }
    return 0;
}