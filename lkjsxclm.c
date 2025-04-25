#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARAM_SIZE (1024 * 1024 * 16)
#define TRAIN_SIZE (1024 * 16)
#define IO_SIZE (1024 * 16)
#define LAYER_SIZE (512)
#define TRAIN_COUNT 100
#define MUTATION_RATE 0.00001
#define RANDOM_SEED 1
#define GEN_COUNT 10

typedef enum {
    RESULT_OK,
    RESULT_ERR,
} result_t;

int8_t param_corrent_data[PARAM_SIZE];
int8_t param_best_data[PARAM_SIZE];
int8_t layer_data[LAYER_SIZE * 2];
char train_data[TRAIN_SIZE];
char io_data[IO_SIZE];

int8_t* layer1 = layer_data;
int8_t* layer2 = layer_data + LAYER_SIZE;

uint64_t rd = RANDOM_SEED;

uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

void param_randomize() {
    for (int i = 0; i < PARAM_SIZE; i++) {
        rd = xorshift64(rd);
        param_corrent_data[i] = (int8_t)(rd % 256);
    }
}

void swap(int8_t** a, int8_t** b) {
    int8_t* tmp = *a;
    *a = *b;
    *b = tmp;
}

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

void cal() {
    int param_i = 0;
    for (int layer2_i = 0; layer2_i < LAYER_SIZE; layer2_i++) {
        layer2[layer2_i] = 0;
        for (int layer1_i = 0; layer1_i < LAYER_SIZE; layer1_i++) {
            layer2[layer2_i] += layer1[layer1_i] * param_corrent_data[param_i++];
        }
        layer2[layer2_i] += param_corrent_data[param_i++];
        layer2[layer2_i] = (layer2[layer2_i] > 0) ? layer2[layer2_i] : 0;
    }
    swap(&layer1, &layer2);
}

void layer_clean() {
    memset(layer1, 0, LAYER_SIZE);
}

void layer_write(char ch) {
    memset(layer1, 0, 256);
    layer1[ch] = 1;
}

char layer_read() {
    int8_t max = -128;
    int8_t out = -128;
    for (int i = 0; i < 256; i++) {
        if (layer1[i] > max) {
            max = layer1[i];
            out = i - 128;
        }
    }
    return out;
}

int64_t score(int8_t correct_index) {
    int64_t score = 0;
    if(layer_read() == correct_index) {
        score += 65536;
    }
    for (int i = 0; i < 256; i++) {
        int8_t a = layer1[i];
        score -= (a < 0) ? -a : a;
    }
    return score;
}

result_t train() {
    int64_t best_score = INT64_MIN;
    for (int train_i = 0; train_i < TRAIN_COUNT; train_i++) {
        int64_t corrent_score = 0;
        memcpy(param_corrent_data, param_best_data, PARAM_SIZE);
        param_randomize();
        layer_clean();
        for (char* train_itr = train_data; *train_itr != '\0'; train_itr++) {
            char ch_in = *train_itr;
            char ch_correct = *(train_itr+1);
            layer_write(ch_in);
            cal();
            putchar(layer_read());
            corrent_score += score(ch_correct);
        }
        if (corrent_score > best_score) {
            best_score = corrent_score;
            memcpy(param_best_data, param_corrent_data, PARAM_SIZE);
            printf("New best score: %lld\n", best_score);
        } else {
            printf("Score did not improve: %lld\n", corrent_score);
        }
    }
    return RESULT_OK;
}

result_t gen() {
    memcpy(param_corrent_data, param_best_data, PARAM_SIZE);
    layer_clean();
    for (char* io_itr = io_data; *io_itr != '\0'; io_itr++) {
        char ch = *io_itr;
        layer_write(ch);
        cal();
    }
    for (int i = 0; i < GEN_COUNT; i++) {
        memset(layer1, 0, 256);
        cal();
        int ch = layer_read();
        printf("Generated char: %d\n", ch);
    }
}

int main() {
    if (readfile(train_data, "./train.txt") != RESULT_OK) {
        return 1;
    }
    if (readfile(io_data, "./input.txt") != RESULT_OK) {
        return 1;
    }
    if (train() != RESULT_OK) {
        return 1;
    }
    if (gen() != RESULT_OK) {
        return 1;
    }
    return 0;
}