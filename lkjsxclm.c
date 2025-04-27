#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LAYERIO_BYTESIZE 256
#define MEMORY_BYTESIZE 256
#define MUTATION_RATE 0.0004
#define THREAD_COUNT 14
#define TRAIN_COUNT 10000
#define OUTPUT_COUNT 1000
#define RANDOM_SEED 123

#define TEXT_BYTESIZE (1024 * 1024)
#define TRAIN_PATH "./train.txt"
#define INPUT_PATH "./input.txt"
#define OUTPUT_PATH "./output.txt"

#define LAYER_BYTESIZE (LAYERIO_BYTESIZE + MEMORY_BYTESIZE)
#define PARAM_BYTESIZE (LAYER_BYTESIZE * LAYER_BYTESIZE * 2)

#define LAYER_BITSIZE (LAYER_BYTESIZE * 8)
#define PARAM_BITSIZE (PARAM_BYTESIZE * 8)

typedef union {
    int64_t i64;
    uint64_t u64;
    uint8_t u8[8];
    int8_t i8[8];
} uni64_t;

uint64_t best_param[PARAM_BYTESIZE / sizeof(uint64_t)];
int64_t best_score = INT64_MIN;
pthread_mutex_t best_param_mutex;

uint8_t train_data[TEXT_BYTESIZE];
uint8_t input_data[TEXT_BYTESIZE];
uint8_t output_data[TEXT_BYTESIZE];

int64_t train_size;
int64_t input_size;
volatile int global_iterations = 0;
pthread_mutex_t print_mutex;

typedef struct {
    int id;
    uint64_t* layer1_u64;
    uint64_t* layer2_u64;
    uint8_t* layer1_u8;
    uint8_t* layer2_u8;
    int8_t* param_i8;
    uint64_t rd;
    uint64_t layer_data[LAYER_BYTESIZE / sizeof(uint64_t) * 2];
    uint64_t param_u64[PARAM_BYTESIZE / sizeof(uint64_t)];
} thread_data_t;

thread_data_t thread_data[THREAD_COUNT];

uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

size_t file_read(uint8_t* dst, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filename);
        perror("Error details");
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    size_t n = ftell(fp);
    if (n >= TEXT_BYTESIZE) {
        fprintf(stderr, "Warning: File %s exceeds buffer size (%d bytes). Truncating.\n", filename, TEXT_BYTESIZE);
        n = TEXT_BYTESIZE - 1;
    }
    fseek(fp, 0, SEEK_SET);
    size_t read_count = fread(dst, 1, n, fp);
    if (read_count != n && ferror(fp)) {
        fprintf(stderr, "Error reading file %s.\n", filename);
        perror("Error details");
        fclose(fp);
        exit(EXIT_FAILURE);
    } else if (read_count != n && !feof(fp)) {
        fprintf(stderr, "Warning: Failed to read full file content from %s. Read %zu of %zu bytes.\n", filename, read_count, n);
    }

    fclose(fp);
    return read_count;
}

size_t file_write(const uint8_t* src, size_t size, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        perror("Error details");
        return 0;
    }
    size_t written_count = fwrite(src, 1, size, fp);
    if (written_count != size) {
        fprintf(stderr, "Warning: Failed to write full content to %s. Wrote %zu of %zu bytes.\n", filename, written_count, size);
        if (ferror(fp)) {
            perror("Error details");
        }
    }
    fclose(fp);
    return written_count;
}

void layer_reset(thread_data_t* td) {
    memset(td->layer_data, 0, sizeof(td->layer_data));
}

void layer_clean(thread_data_t* td) {
    memset(td->layer1_u8, 0, LAYERIO_BYTESIZE);
}

void layer_setchar(thread_data_t* td, uint8_t index) {
    memset(td->layer1_u8, 0, LAYERIO_BYTESIZE);
    if (index < LAYERIO_BYTESIZE) {
        td->layer1_u8[index] = UINT8_MAX;

    } else {
        fprintf(stderr, "Warning: layer_setchar index %u out of bounds (%d)\n", index, LAYERIO_BYTESIZE);
    }
}

uint8_t layer_getchar(thread_data_t* td) {
    uint8_t max_value = 0;
    uint8_t max_index = 0;

    for (int i = 0; i < LAYERIO_BYTESIZE; i++) {
        if (td->layer1_u8[i] > max_value) {
            max_value = td->layer1_u8[i];
            max_index = i;
        }
    }

    return max_index;
}

int64_t layer_score(thread_data_t* td, uint8_t correct_ch) {
    int64_t score = 0;
    const int64_t correct_bonus_multiplier = LAYERIO_BYTESIZE / 32;
    const int64_t perfect_match_bonus = LAYERIO_BYTESIZE * UINT8_MAX / 32;

    for (int i = 0; i < LAYERIO_BYTESIZE; i++) {
        score -= td->layer1_u8[i];
    }

    score += (int64_t)td->layer1_u8[correct_ch] * correct_bonus_multiplier;

    if (layer_getchar(td) == correct_ch) {
        score += perfect_match_bonus;
    }

    return score;
}

void layer_cal(thread_data_t* td) {
    int param_i = 0;
    for (int dst_i = 0; dst_i < LAYER_BYTESIZE; dst_i++) {
        int64_t sum = td->param_i8[param_i++];
        for (int src_i = 0; src_i < LAYER_BYTESIZE; src_i++) {
            sum += td->layer1_u8[src_i] * td->param_i8[param_i++];
        }
        sum /= 2048;
        if (sum < 0) {
            sum = 0;
        } else if (sum > UINT8_MAX) {
            sum = UINT8_MAX;
        }
        sum |= sum >> 4;
        sum |= sum >> 2;
        sum |= sum >> 1;
        td->layer2_u8[dst_i] = sum;
    }

    uint64_t* tmp_u64 = td->layer1_u64;
    td->layer1_u64 = td->layer2_u64;
    td->layer2_u64 = tmp_u64;

    uint8_t* tmp_u8 = td->layer1_u8;
    td->layer1_u8 = td->layer2_u8;
    td->layer2_u8 = tmp_u8;
}

void param_init(uint64_t seed) {
    uint64_t current_rd = seed;
    printf("Initializing global best parameters with seed %llu...\n", (unsigned long long)seed);
    for (int i = 0; i < PARAM_BYTESIZE / sizeof(uint64_t); i++) {
        current_rd = xorshift64(current_rd);
        best_param[i] = current_rd;
    }
    printf("Global parameters initialized.\n");

    printf("Initializing thread-local data...\n");
    for (int i = 0; i < THREAD_COUNT; ++i) {
        thread_data[i].id = i;

        thread_data[i].layer1_u64 = thread_data[i].layer_data;
        thread_data[i].layer2_u64 = thread_data[i].layer_data + (LAYER_BITSIZE / 64);
        thread_data[i].layer1_u8 = (uint8_t*)(thread_data[i].layer1_u64);
        thread_data[i].layer2_u8 = (uint8_t*)(thread_data[i].layer2_u64);
        thread_data[i].param_i8 = (int8_t*)(thread_data[i].param_u64);
        thread_data[i].rd = RANDOM_SEED + thread_data[i].id;
    }
    printf("Thread data initialized.\n");
}

void param_mutate(thread_data_t* td) {
    pthread_mutex_lock(&best_param_mutex);
    memcpy(td->param_u64, best_param, sizeof(best_param));
    pthread_mutex_unlock(&best_param_mutex);

    int mutations = 0;

    uint64_t threshold = (uint64_t)(UINT64_MAX * MUTATION_RATE);
    if (threshold == 0 && MUTATION_RATE > 0.0) {
        threshold = 1;
    }

    for (int i = 0; i < PARAM_BYTESIZE / sizeof(uint64_t); i++) {
        td->rd = xorshift64(td->rd);
        if (td->rd < threshold) {
            td->rd = xorshift64(td->rd);
            td->param_u64[i] = td->rd;
            mutations++;
        }
    }
}

int64_t evaluate(thread_data_t* td, const uint64_t* current_param) {
    int64_t current_total_score = 0;
    uint8_t ch_out;
    uint8_t ch_in;
    uint8_t ch_correct;

    layer_reset(td);

    for (int64_t i = 0; i < train_size / 2; i++) {
        ch_in = train_data[i];
        ch_correct = train_data[i + 1];

        layer_setchar(td, ch_in);
        layer_cal(td);

        current_total_score += layer_score(td, ch_correct);

        ch_out = layer_getchar(td);
        if (ch_out != ch_correct) {
            return current_total_score;
        }
    }

    for (int64_t i = train_size / 2; i < train_size - 1; i++) {
        ch_correct = train_data[i + 1];

        layer_setchar(td, ch_in);
        layer_cal(td);
        ch_out = layer_getchar(td);

        current_total_score += layer_score(td, ch_correct);

        if (ch_out != ch_correct) {
            return current_total_score;
        }
        ch_in = ch_out;
    }

    return current_total_score;
}