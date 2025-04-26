#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IO_BYTESIZE 256
#define MEMORY_BYTESIZE 256
#define LAYER_BITSIZE ((IO_BYTESIZE + MEMORY_BYTESIZE) * 8)
#define PARAM_BITSIZE (LAYER_BITSIZE * LAYER_BITSIZE * 2)
#define TRAIN_BITSIZE (1024 * 1024 * 8)
#define RANDOM_SEED 1

uint64_t best_param[PARAM_BITSIZE / 64];
uint8_t train1[TRAIN_BITSIZE / 8];
uint8_t train2[TRAIN_BITSIZE / 8];
int64_t best_score = 0;

uint64_t param[PARAM_BITSIZE / 64];
uint64_t layer_data[LAYER_BITSIZE / 64 * 2];
uint64_t* layer1_u64 = layer_data;
uint64_t* layer2_u64 = layer_data + LAYER_BITSIZE / 64;
uint8_t* layer1_u8 = (uint8_t*)(layer_data);
uint8_t* layer2_u8 = (uint8_t*)(layer_data + LAYER_BITSIZE / 64);
uint64_t rd = RANDOM_SEED;

uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

void layer_clean() {
    memset(layer1_u8, 0, LAYER_BITSIZE / 8);
}

void layer_write(uint8_t index) {
    memset(layer1_u8, 0, IO_BYTESIZE);
    layer1_u8[index] = UINT8_MAX;
}

uint8_t layer_read(uint8_t index) {
    uint8_t max_value = 0;
    uint8_t max_index = 0;
    for (int i = 0; i < 256; i++) {
        if (layer1_u8[i] > max_value) {
            max_value = layer1_u8[i];
            max_index = i;
        }
    }
    return max_index;
}

void layer_cal() {
    int param_i = 0;
    for (int output_i = 0; output_i < LAYER_BITSIZE / 8; output_i++) {
        int64_t x = 0;
        for (int input_i = 0; input_i < LAYER_BITSIZE / 64; input_i++) {
            x += layer1_u64[input_i] & param[param_i++];
            x -= layer1_u64[input_i] & param[param_i++];
        }
        layer2_u8[output_i] = x;
    }
    uint64_t* tmp = layer1_u64;
    layer1_u64 = layer2_u64;
    layer2_u64 = tmp;
}

int64_t score(uint8_t correct_ch) {
    int64_t score = 0;
    for (int i = 0; i < LAYER_BITSIZE / 8; i++) {
        score -= layer1_u8[i];
    }
    score += layer1_u8[correct_ch] * 256;
    return score;
}