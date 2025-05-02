#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define LAYERIO_BYTESIZE 256
#define MEMORY_BYTESIZE (1024 - LAYERIO_BYTESIZE)
#define MUTATION_RATE 0.0001
#define PRUNING_LIMIT 0
#define THINK_LIMIT 6
#define THINK_REWARD 32

#define SCORE_REWARD_CORRECT_ACTIVATION (LAYERIO_BYTESIZE / 32)
#define SCORE_REWARD_THINK_ACTIVATION (LAYERIO_BYTESIZE / 64)
#define SCORE_BONUS_EXACT_MATCH (LAYERIO_BYTESIZE * UINT8_MAX / 32)

#define THREAD_COUNT 14
#define TRAIN_COUNT 200000
#define OUTPUT_COUNT 1000
#define RANDOM_SEED 123

#define TEXT_BYTESIZE (1024 * 1024 * 16)
#define TEXT_SETSIZE 1024
#define TRAIN_PATH "./train.txt"
#define INPUT_PATH "./input.txt"
#define OUTPUT_PATH "./output.txt"
#define PARAM_PATH "./param.bin"

#define THINK_TOKEN '\t'
#define LAYER_BYTESIZE (LAYERIO_BYTESIZE + MEMORY_BYTESIZE)
#define PARAM_BYTESIZE (LAYER_BYTESIZE * (LAYER_BYTESIZE + 1))
#define THREAD_TRAIN_COUNT (TRAIN_COUNT / THREAD_COUNT)
#define LAYER_CAL_SCALE_SHIFT 11
#define SCORE_PENALTY_PER_ACTIVATION 1

typedef struct {
    const uint8_t* data;
    size_t size;
} string_t;

typedef struct {
    int64_t tid;
    uint8_t* layer1_u8;
    uint8_t* layer2_u8;
    int8_t* param_i8;
    uint64_t rd;

    uint8_t layer_data[LAYER_BYTESIZE * 2];

    uint8_t param_bytes[PARAM_BYTESIZE];

} thread_t;

typedef struct {
    int64_t best_score;
    size_t train_size;
    size_t input_size;
    size_t train_set_count;

    uint8_t train_data[TEXT_BYTESIZE];
    uint8_t input_data[TEXT_BYTESIZE];
    uint8_t output_data[TEXT_BYTESIZE];

    string_t train_set[TEXT_SETSIZE * 2];

    uint8_t best_param[PARAM_BYTESIZE];

    pthread_mutex_t best_param_mutex;

} global_t;

global_t global_data;
thread_t thread_data[THREAD_COUNT];
volatile bool keep_running = true;

uint64_t xorshift64(uint64_t x);
size_t file_read(uint8_t* dst, const char* filename, size_t max_size);
size_t file_write(const uint8_t* src, size_t size, const char* filename);
void initialize_parameters(uint8_t* params, size_t size, uint64_t* seed);
void param_update(thread_t* td);
void layer_reset(thread_t* td);

void layer_setchar(thread_t* td, uint8_t index);
uint8_t layer_getchar(thread_t* td);
int64_t layer_score(thread_t* td, uint8_t predicted_char, uint8_t correct_char);
void layer_cal(thread_t* td);
int64_t evaluate(thread_t* td);
void* thread_func(void* arg);
void generate_output(const char* output_filename);
int parse_train_data();

uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

size_t file_read(uint8_t* dst, const char* filename, size_t max_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open file '%s': ", filename);
        perror(NULL);
        return 0;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ERROR: Failed to seek in file '%s': ", filename);
        perror(NULL);
        fclose(fp);
        return 0;
    }
    long file_size_long = ftell(fp);
    if (file_size_long < 0) {
        fprintf(stderr, "ERROR: Failed to get size of file '%s': ", filename);
        perror(NULL);
        fclose(fp);
        return 0;
    }
    size_t file_size = (size_t)file_size_long;

    if (file_size >= max_size) {
        fprintf(stderr, "WARNING: File '%s' (size %zu) exceeds buffer size (%zu). Truncating.\n",
                filename, file_size, max_size);
        file_size = max_size - 1;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: Failed to seek to start of file '%s': ", filename);
        perror(NULL);
        fclose(fp);
        return 0;
    }
    size_t read_count = fread(dst, 1, file_size, fp);

    if (read_count != file_size) {
        if (feof(fp)) {
            fprintf(stderr, "WARNING: Unexpected end of file in '%s'. Read %zu of expected %zu bytes.\n",
                    filename, read_count, file_size);
        } else if (ferror(fp)) {
            fprintf(stderr, "ERROR: Error reading file '%s': ", filename);
            perror(NULL);
            fclose(fp);
            return 0;
        } else {
            fprintf(stderr, "WARNING: Incomplete read from file '%s'. Read %zu of expected %zu bytes.\n",
                    filename, read_count, file_size);
        }
    }

    fclose(fp);

    return read_count;
}

size_t file_write(const uint8_t* src, size_t size, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open file '%s' for writing: ", filename);
        perror(NULL);
        return 0;
    }

    size_t written_count = fwrite(src, 1, size, fp);

    if (written_count != size) {
        fprintf(stderr, "WARNING: Failed to write full content to '%s'. Wrote %zu of %zu bytes.\n",
                filename, written_count, size);
        if (ferror(fp)) {
            fprintf(stderr, "ERROR: Error writing to file '%s': ", filename);
            perror(NULL);
        }
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "ERROR: Failed to close file '%s' after writing: ", filename);
        perror(NULL);
    }
    return written_count;
}

void initialize_parameters(uint8_t* params, size_t size, uint64_t* seed) {
    printf("Initializing %zu parameter bytes with random values...\n", size);
    for (size_t i = 0; i < size; ++i) {
        *seed = xorshift64(*seed);
        params[i] = (int8_t)((*seed) & 0xFF);
    }
}

void param_update(thread_t* td) {
    pthread_mutex_lock(&global_data.best_param_mutex);
    memcpy(td->param_bytes, global_data.best_param, PARAM_BYTESIZE);
    pthread_mutex_unlock(&global_data.best_param_mutex);

    const uint64_t mutation_threshold = (uint64_t)(MUTATION_RATE * UINT64_MAX);

    for (size_t i = 0; i < PARAM_BYTESIZE; i++) {
        td->rd = xorshift64(td->rd);
        if (td->rd < mutation_threshold * 8) {
            td->rd = xorshift64(td->rd);
            td->param_bytes[i] ^= (uint8_t)(1 << (td->rd % 8));
        }
    }
}

void layer_reset(thread_t* td) {
    memset(td->layer_data, 0, sizeof(td->layer_data));

    td->layer1_u8 = td->layer_data;
    td->layer2_u8 = td->layer_data + LAYER_BYTESIZE;
}

void layer_setchar(thread_t* td, uint8_t index) {
    memset(td->layer1_u8, 0, LAYERIO_BYTESIZE);

    if (index < LAYERIO_BYTESIZE) {
        td->layer1_u8[index] = UINT8_MAX;
    } else {
        fprintf(stderr, "Warning: layer_setchar index %u out of bounds (%d), setting index 0\n", index, LAYERIO_BYTESIZE);
        td->layer1_u8[0] = UINT8_MAX;
    }
}

uint8_t layer_getchar(thread_t* td) {
    uint8_t max_value = 0;
    uint8_t max_index = 0;

    for (size_t i = 0; i < LAYERIO_BYTESIZE; i++) {
        if (td->layer1_u8[i] > max_value) {
            max_value = td->layer1_u8[i];
            max_index = (uint8_t)i;
        }
    }
    return max_index;
}

int64_t layer_score(thread_t* td, uint8_t predicted_char, uint8_t correct_char) {
    int64_t score = 0;

    for (size_t i = 0; i < LAYERIO_BYTESIZE; i++) {
        score += (UINT8_MAX - td->layer1_u8[i]) * SCORE_PENALTY_PER_ACTIVATION;
    }

    int64_t popcount = 0;
    int64_t prefferd_popcount = (LAYER_BYTESIZE - LAYERIO_BYTESIZE) * 8 / 2;
    for (size_t i = LAYERIO_BYTESIZE / 8; i < LAYER_BYTESIZE / 8; i++) {
        popcount += __builtin_popcountll((uint64_t)(td->layer1_u8)[i]);
    }
    score += (prefferd_popcount - abs(prefferd_popcount - popcount)) / 4;


    score += (int64_t)td->layer1_u8[correct_char] * SCORE_REWARD_CORRECT_ACTIVATION;
    score += (int64_t)td->layer1_u8[THINK_TOKEN] * SCORE_REWARD_THINK_ACTIVATION;

    if (predicted_char == correct_char) {
        score += SCORE_BONUS_EXACT_MATCH;
    }

    return score;
}

void layer_cal(thread_t* td) {
    const int8_t* params = td->param_i8;
    int param_idx = 0;

    for (size_t dst_i = 0; dst_i < LAYER_BYTESIZE; dst_i++) {
        int64_t sum = params[param_idx++];

        for (size_t src_i = 0; src_i < LAYER_BYTESIZE; src_i++) {
            sum += (int64_t)td->layer1_u8[src_i] * (int64_t)params[param_idx++];
        }

        if (sum < 0) {
            sum = 0;
        }

        sum >>= LAYER_CAL_SCALE_SHIFT;

        if (sum > UINT8_MAX) {
            sum = UINT8_MAX;
        }

        td->layer2_u8[dst_i] = (uint8_t)sum;
    }

    uint8_t* tmp_u8 = td->layer1_u8;
    td->layer1_u8 = td->layer2_u8;
    td->layer2_u8 = tmp_u8;
}

int64_t evaluate(thread_t* td) {
    int64_t total_score = 0;
    int64_t total_correct_count = 0;
    int64_t total_wrong_count = 0;

    for (size_t pair_idx = 0; pair_idx < global_data.train_set_count; ++pair_idx) {
        const string_t* prompt_line = &global_data.train_set[pair_idx * 2];
        const string_t* response_line = &global_data.train_set[pair_idx * 2 + 1];

        layer_reset(td);
        int64_t current_wrong_streak = 0;
        int64_t current_think_streak = 0;
        uint8_t ch_in;
        uint8_t ch_correct;
        uint8_t ch_predicted;

        for (size_t i = 0; i < prompt_line->size; ++i) {
            ch_in = prompt_line->data[i];
            layer_setchar(td, ch_in);
            layer_cal(td);
        }

        for (size_t i = 0; i < response_line->size; ++i) {
            ch_correct = response_line->data[i];
            ch_predicted = layer_getchar(td);

            int64_t step_score = layer_score(td, ch_predicted, ch_correct);
            total_score += step_score;

            if (ch_predicted == ch_correct) {
                total_correct_count++;
                current_wrong_streak = 0;
                current_think_streak = 0;
            } else if (ch_predicted == THINK_TOKEN) {
                current_think_streak++;
                if (current_think_streak <= THINK_LIMIT) {
                    total_score += THINK_REWARD;
                    i--;

                    continue;
                } else {
                    total_wrong_count++;
                    current_wrong_streak++;
                    current_think_streak = 0;
                }
            } else {
                total_wrong_count++;
                current_wrong_streak++;
                current_think_streak = 0;
            }

            if (current_wrong_streak > PRUNING_LIMIT) {
                break;
            }

            ch_in = ch_correct;

            layer_setchar(td, ch_in);

            layer_cal(td);
        }
    }

    return total_score;
}

void* thread_func(void* arg) {
    thread_t* td = (thread_t*)arg;

    for (size_t train_step = 0; train_step < THREAD_TRAIN_COUNT; ++train_step) {
        param_update(td);

        int64_t current_score = evaluate(td);

        if (current_score > global_data.best_score) {
            pthread_mutex_lock(&global_data.best_param_mutex);
            global_data.best_score = current_score;

            memcpy(global_data.best_param, td->param_bytes, PARAM_BYTESIZE);

            if (train_step % (THREAD_TRAIN_COUNT / 10) == 0 || current_score > global_data.best_score - 1000)
                printf("T:%2lld, Progression:%3d, New best score: %lld\n", td->tid, (100 * train_step) / THREAD_TRAIN_COUNT, current_score);
            pthread_mutex_unlock(&global_data.best_param_mutex);
        }
    }

    return NULL;
}

void generate_output(const char* output_filename) {
    printf("\n--- Generating Output ---\n");
    printf("Using best parameters found during training (Score: %lld).\n", global_data.best_score);

    thread_t* td = &thread_data[0];

    memcpy(td->param_bytes, global_data.best_param, PARAM_BYTESIZE);
    td->param_i8 = (int8_t*)td->param_bytes;

    layer_reset(td);

    uint8_t current_char;

    printf("Priming model with input sequence (%zu bytes from %s)...\n", global_data.input_size, INPUT_PATH);
    if (global_data.input_size == 0) {
        printf("Warning: Input file is empty. Starting generation from zero state.\n");
        current_char = 0;
    } else {
        for (size_t i = 0; i < global_data.input_size; ++i) {
            current_char = global_data.input_data[i];
            layer_setchar(td, current_char);
            layer_cal(td);
        }

        current_char = layer_getchar(td);
    }

    printf("Generating %d characters...\n", OUTPUT_COUNT);
    size_t output_idx = 0;
    int think_streak = 0;
    while (output_idx < OUTPUT_COUNT && think_streak < THINK_LIMIT * 2) {
        if (current_char != THINK_TOKEN) {
            global_data.output_data[output_idx++] = current_char;
            think_streak = 0;
        } else {
            think_streak++;

            printf("Warning: Generated THINK_TOKEN (streak %d)\n", think_streak);
        }

        layer_setchar(td, current_char);

        layer_cal(td);

        current_char = layer_getchar(td);
    }

    printf("Writing %zu generated characters to %s...\n", output_idx, output_filename);
    size_t written = file_write(global_data.output_data, output_idx, output_filename);
    if (written == output_idx) {
        printf("Successfully wrote %zu characters.\n", written);
    } else {
        fprintf(stderr, "ERROR: Failed to write the full generated output to %s (wrote %zu).\n", output_filename, written);
    }
}

int parse_train_data() {
    printf("Parsing training data into line pairs...\n");
    const uint8_t* start = global_data.train_data;
    const uint8_t* end = global_data.train_data + global_data.train_size;
    const uint8_t* current = start;
    size_t line_index = 0;

    while (current < end && line_index < TEXT_SETSIZE * 2) {
        const uint8_t* line_start = current;
        const uint8_t* newline = memchr(current, '\n', end - current);

        size_t line_len;
        if (newline) {
            line_len = newline - line_start;
            current = newline + 1;
        } else {
            line_len = end - line_start;
            current = end;
        }

        global_data.train_set[line_index].data = line_start;
        global_data.train_set[line_index].size = line_len;
        line_index++;

        if (current >= end)
            break;
    }

    if (line_index >= TEXT_SETSIZE * 2 && current < end) {
        fprintf(stderr, "Warning: Exceeded maximum training lines (%d). Truncating data.\n", TEXT_SETSIZE);
    }

    if (line_index % 2 != 0) {
        fprintf(stderr, "ERROR: Training data contains an odd number of lines (%zu). Expected pairs.\n", line_index);
        return -1;
    }

    global_data.train_set_count = line_index / 2;
    printf("Parsed %zu training line pairs.\n", global_data.train_set_count);

    if (global_data.train_set_count == 0) {
        fprintf(stderr, "ERROR: No valid training line pairs found in %s.\n", TRAIN_PATH);
        return -1;
    }

    return 0;
}

int main() {
    printf("--- Language Model Training & Generation (C) ---\n");
    printf("Configuration:\n");
    printf("  Layer IO Size: %d, Memory Size: %d, Total Layer: %d\n", LAYERIO_BYTESIZE, MEMORY_BYTESIZE, LAYER_BYTESIZE);
    printf("  Param Size: %d bytes\n", PARAM_BYTESIZE);
    printf("  Mutation Rate: %f, Pruning Limit: %d\n", MUTATION_RATE, PRUNING_LIMIT);
    printf("  Threads: %d, Train Cycles (Total): %d, Cycles/Thread: %d\n", THREAD_COUNT, TRAIN_COUNT, THREAD_TRAIN_COUNT);
    printf("  Output Length: %d, Random Seed: %d\n", OUTPUT_COUNT, RANDOM_SEED);
    printf("------------------------------------------------\n");

    printf("Initializing...\n");

    global_data.best_score = LLONG_MIN;
    global_data.train_size = 0;
    global_data.input_size = 0;
    global_data.train_set_count = 0;
    memset(global_data.best_param, 0, PARAM_BYTESIZE);

    uint64_t master_seed = (RANDOM_SEED == 0) ? (uint64_t)time(NULL) : (uint64_t)RANDOM_SEED;
    master_seed = xorshift64(master_seed);
    if (master_seed == 0)
        master_seed = 1;

    initialize_parameters(global_data.best_param, PARAM_BYTESIZE, &master_seed);

    if (pthread_mutex_init(&global_data.best_param_mutex, NULL) != 0) {
        perror("Mutex initialization failed");
        return EXIT_FAILURE;
    }

    printf("Loading training data from %s...\n", TRAIN_PATH);
    global_data.train_size = file_read(global_data.train_data, TRAIN_PATH, TEXT_BYTESIZE);
    if (global_data.train_size == 0) {
        fprintf(stderr, "ERROR: Failed to read or empty training data file: %s\n", TRAIN_PATH);
        pthread_mutex_destroy(&global_data.best_param_mutex);
        return EXIT_FAILURE;
    }
    printf("Loaded %zu bytes of training data.\n", global_data.train_size);

    if (parse_train_data() != 0) {
        pthread_mutex_destroy(&global_data.best_param_mutex);
        return EXIT_FAILURE;
    }

    printf("Loading input data from %s...\n", INPUT_PATH);
    global_data.input_size = file_read(global_data.input_data, INPUT_PATH, TEXT_BYTESIZE);

    printf("Loaded %zu bytes of input data.\n", global_data.input_size);

    printf("Initializing %d threads...\n", THREAD_COUNT);
    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        thread_data[i].tid = i;

        master_seed = xorshift64(master_seed);
        thread_data[i].rd = (master_seed == 0) ? 1 : master_seed;

        thread_data[i].layer1_u8 = thread_data[i].layer_data;
        thread_data[i].layer2_u8 = thread_data[i].layer_data + LAYER_BYTESIZE;

        thread_data[i].param_i8 = (int8_t*)thread_data[i].param_bytes;
    }

    printf("\n--- Starting Training Phase ---\n");
    pthread_t threads[THREAD_COUNT];
    time_t start_time = time(NULL);

    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        if (pthread_create(&threads[i], NULL, thread_func, &thread_data[i]) != 0) {
            perror("Failed to create thread");

            keep_running = false;
            pthread_mutex_destroy(&global_data.best_param_mutex);
            return EXIT_FAILURE;
        }
    }

    printf("Waiting for %d threads to complete %d cycles each...\n", THREAD_COUNT, THREAD_TRAIN_COUNT);
    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("Failed to join thread");
        }
    }

    time_t end_time = time(NULL);
    double duration = difftime(end_time, start_time);
    printf("\n--- Training Phase Completed ---\n");
    printf("Duration: %.2f seconds\n", duration);
    printf("Final best score found: %lld\n", global_data.best_score);

    pthread_mutex_destroy(&global_data.best_param_mutex);

    generate_output(OUTPUT_PATH);

    printf("\n--- Program Finished ---\n");
    return EXIT_SUCCESS;
}