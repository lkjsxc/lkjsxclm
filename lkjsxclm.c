#include <limits.h>
#include <pthread.h>  // Include pthread
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // For seeding random numbers per thread
#include <stdbool.h> // For bool type

#define LAYERIO_BYTESIZE 256
#define MEMORY_BYTESIZE 256
#define MUTATION_RATE 0.01
#define THREAD_COUNT 12
#define TRAIN_COUNT 100000  // Increased train count to see parallel benefits
#define OUTPUT_COUNT 1000
#define RANDOM_SEED 1

#define TEXT_BITSIZE (1024 * 1024 * 8)
#define TRAIN_PATH "./train.txt"
#define INPUT_PATH "./input.txt"
#define OUTPUT_PATH "./output.txt"

#define LAYER_BYTESIZE (LAYERIO_BYTESIZE + MEMORY_BYTESIZE) // 512 Bytes
#define LAYER_BITSIZE (LAYER_BYTESIZE * 8)                // 4096 Bits

// Parameter matrix size: Each output bit depends on each input bit.
// Two matrices: one for setting bits (0->1), one for clearing bits (1->0).
#define PARAM_BITSIZE (LAYER_BITSIZE * LAYER_BITSIZE * 2)
#define PARAM_U64_SIZE (PARAM_BITSIZE / 64) // Total uint64_t needed for parameters

typedef union {
    int64_t i64;
    uint64_t u64;
    uint8_t u8[8];
    int8_t i8[8];
} uni64_t;

// --- Global Shared Data ---
uint64_t best_param[PARAM_U64_SIZE];
int64_t best_score = INT64_MIN;
pthread_mutex_t best_param_mutex;  // Mutex for best_param and best_score

uint8_t train_data[TEXT_BITSIZE / 8];
uint8_t input_data[TEXT_BITSIZE / 8];
uint8_t output_data[TEXT_BITSIZE / 8];

int64_t train_size;
int64_t input_size;
volatile int global_iterations = 0;  // Track total iterations across threads
pthread_mutex_t print_mutex;         // Mutex for coordinated printing

// --- Thread Local Data Structure ---
typedef struct {
    int id;                                // Thread ID for debugging/seeding
    uint64_t local_param[PARAM_U64_SIZE];  // Thread's copy for mutation
    // Allocate double the layer size for swapping pointers efficiently
    uint64_t layer_data[LAYER_BITSIZE / 64 * 2];
    uint64_t* layer1_u64;  // Pointer to current input layer (as uint64_t)
    uint64_t* layer2_u64;  // Pointer to computed output layer (as uint64_t)
    uint8_t* layer1_u8;    // Pointer to current input layer (as uint8_t)
    uint8_t* layer2_u8;    // Pointer to computed output layer (as uint8_t)
    uint64_t rd;           // Thread-local random seed state for xorshift64
} thread_data_t;

thread_data_t thread_data[THREAD_COUNT];

// --- Utility Functions ---
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
    if (n >= TEXT_BITSIZE / 8) {
        fprintf(stderr, "Warning: File %s exceeds buffer size (%d bytes). Truncating.\n", filename, TEXT_BITSIZE / 8);
        n = TEXT_BITSIZE / 8 - 1;  // Leave space for null terminator if needed? Original didn't.
    }
    fseek(fp, 0, SEEK_SET);
    size_t read_count = fread(dst, 1, n, fp);
    if (read_count != n && ferror(fp)) {  // Check ferror
        fprintf(stderr, "Error reading file %s.\n", filename);
        perror("Error details");
        fclose(fp);
        exit(EXIT_FAILURE);  // Exit on read error
    } else if (read_count != n && !feof(fp)) { // Check if read stopped before EOF without error
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

// --- Layer Functions (Operate on Thread Data) ---

void layer_clean(thread_data_t* td) {
    // Cleans only the IO part, memory part persists between calls
    memset(td->layer1_u8, 0, LAYERIO_BYTESIZE);
    // Note: layer2 is the destination, it gets overwritten or zeroed in layer_cal
}

// Cleans the entire layer state (IO + Memory) for both buffers
void layer_reset(thread_data_t* td) {
    memset(td->layer_data, 0, sizeof(td->layer_data));
    // Pointers remain valid, pointing to different parts of the zeroed buffer
}

void layer_setchar(thread_data_t* td, uint8_t index) {
    // Clean IO part only before setting the new character
    memset(td->layer1_u8, 0, LAYERIO_BYTESIZE);
    if (index < LAYERIO_BYTESIZE) {              // Bounds check
        // Set the corresponding byte to maximum activation
        td->layer1_u8[index] = UINT8_MAX;
        // Alternative: Set a single bit? The scoring suggests byte values matter.
        // uint64_t bit_index = index * 8; // If index represents a byte
        // td->layer1_u64[bit_index / 64] |= (1ULL << (bit_index % 64));
    } else {
        // This case should ideally not happen if input chars are 0-255 and LAYERIO_BYTESIZE is 256
        fprintf(stderr, "Warning: layer_setchar index %u out of bounds (%d)\n", index, LAYERIO_BYTESIZE);
    }
}

uint8_t layer_getchar(thread_data_t* td) {
    uint8_t max_value = 0;
    uint8_t max_index = 0; // Default to 0 if all are 0

    // Only check the IO part of the layer for the output character
    for (int i = 0; i < LAYERIO_BYTESIZE; i++) {
        if (td->layer1_u8[i] > max_value) {
            max_value = td->layer1_u8[i];
            max_index = i;
        }
    }
    // If max_value is still 0, it means no output neuron was activated significantly.
    // Returning max_index (which is 0) is a reasonable default.
    return max_index;
}

int64_t layer_score(thread_data_t* td, uint8_t correct_ch) {
    int64_t score = 0;
    const int64_t correct_bonus_multiplier = LAYERIO_BYTESIZE / 32; // 8
    const int64_t perfect_match_bonus = LAYERIO_BYTESIZE * UINT8_MAX / 32; // ~65k

    // Iterate through the IO part of the layer
    for (int i = 0; i < LAYERIO_BYTESIZE; i++) {
        // General penalty for any activation (encourages sparse output)
        score -= td->layer1_u8[i];
    }

    // Add back the penalty for the correct char and add a bonus proportional to its activation
    // score = score - (-td->layer1_u8[correct_ch]) + (td->layer1_u8[correct_ch] * bonus_multiplier)
    // score = score + td->layer1_u8[correct_ch] + (td->layer1_u8[correct_ch] * bonus_multiplier)
    // score = score + td->layer1_u8[correct_ch] * (1 + bonus_multiplier)
    // Original calculation: score += td->layer1_u8[correct_ch] * LAYERIO_BYTESIZE / 32;
    // This implies the penalty was added first, let's stick to that logic:
    score += (int64_t)td->layer1_u8[correct_ch] * correct_bonus_multiplier;

    // Large bonus if the character with the highest activation is the correct one
    if (layer_getchar(td) == correct_ch) {
        // Add a substantial bonus for getting the prediction right
        score += perfect_match_bonus;
    }

    return score;
}

/**
 * @brief Calculates the next layer state based on the current state and parameters.
 *
 * This function implements the core transformation logic of the layer. It uses
 * two large parameter matrices (derived from current_param) to determine how
 * the input bits influence the output bits. One matrix controls setting bits (0->1)
 * and the other controls clearing bits (1->0).
 *
 * The calculation follows these steps:
 * 1. Identify the source (input) and destination (output) buffers based on the
 *    current pointers in thread_data_t (td).
 * 2. Initialize the destination buffer to all zeros.
 * 3. Initialize a 'clear_mask' buffer to all ones.
 * 4. Iterate through each bit 'i' of the source (input) layer.
 * 5. If input bit 'i' is set (1):
 *    a. Iterate through each potential output bit 'j'.
 *    b. Check the corresponding parameter bit in the 'set' matrix (P_set[j][i]).
 *       If P_set[j][i] is 1, set the j-th bit in the destination buffer (using OR).
 *    c. Check the corresponding parameter bit in the 'clear' matrix (P_clear[j][i]).
 *       If P_clear[j][i] is 1, clear the j-th bit in the 'clear_mask' buffer (using AND NOT).
 * 6. After checking all input bits, apply the 'clear_mask' to the destination
 *    buffer using a bitwise AND operation. This ensures bits marked for clearing
 *    are forced to zero, regardless of whether they were also marked for setting.
 * 7. Finally, swap the input and output pointers in 'td' so the newly calculated
 *    state becomes the input for the *next* call to layer_cal.
 *
 * @warning Performance: This implementation performs approximately
 *          LAYER_BITSIZE * LAYER_BITSIZE bit checks and operations per call.
 *          For the default LAYER_BITSIZE = 4096, this is extremely computationally
 *          expensive (~16.7 million inner loop iterations). Performance may be
 *          a major bottleneck. Consider optimizing with SIMD, different data
 *          structures, or hardware acceleration if needed.
 *
 * @param td Pointer to the thread's local data, containing layer buffers.
 * @param current_param Pointer to the parameter set (containing both set/clear matrices)
 *                      to use for the calculation.
 */
void layer_cal(thread_data_t* td, const uint64_t* current_param) {
    // --- Step 1: Identify Source and Destination Buffers ---
    // Before the calculation, layer1 holds the *current* state (input for this step)
    // and layer2 is the buffer where the *next* state (output) will be calculated.
    const uint64_t* input_layer_u64 = td->layer1_u64;
    uint64_t* output_layer_u64 = td->layer2_u64; // Destination buffer

    // --- Step 2 & 3: Initialize Output and Clear Mask ---
    memset(output_layer_u64, 0, LAYER_BYTESIZE); // Clear destination buffer

    // Temporary buffer to accumulate bits that should be cleared. Start with all ones.
    // Using stack allocation assuming LAYER_BYTESIZE is manageable. If very large,
    // consider allocating within thread_data_t or dynamically.
    uint64_t clear_mask_u64[LAYER_BITSIZE / 64];
    memset(clear_mask_u64, 0xFF, LAYER_BYTESIZE); // Initialize mask to all ones

    // Base index offset in current_param for the clear matrix (starts after the set matrix)
    const size_t param_clear_base_offset_u64 = PARAM_BITSIZE / 2 / 64; // PARAM_U64_SIZE / 2

    // --- Step 4 & 5: Iterate through Input Bits and Apply Parameters ---
    for (uint64_t i = 0; i < LAYER_BITSIZE; ++i) {
        // Check if input bit 'i' is set
        uint64_t i_word_idx = i / 64;
        uint64_t i_bit_offset = i % 64;
        bool input_bit_is_set = (input_layer_u64[i_word_idx] >> i_bit_offset) & 1ULL;

        if (input_bit_is_set) {
            // Input bit 'i' is 1, so it can influence output bits 'j'
            // based on the i-th column of the parameter matrices.
            for (uint64_t j = 0; j < LAYER_BITSIZE; ++j) {
                uint64_t j_word_idx = j / 64;
                uint64_t j_bit_offset = j % 64;

                // --- 5.b: Check Set Parameter P_set[j][i] ---
                uint64_t set_param_flat_idx = j * LAYER_BITSIZE + i; // Flattened index [row][col]
                uint64_t set_param_word_idx = set_param_flat_idx / 64;
                uint64_t set_param_bit_offset = set_param_flat_idx % 64;

                // Check if the parameter bit allowing input 'i' to set output 'j' is active
                bool set_param_is_set = (current_param[set_param_word_idx] >> set_param_bit_offset) & 1ULL;

                if (set_param_is_set) {
                    // OR the j-th bit into the output buffer
                    output_layer_u64[j_word_idx] |= (1ULL << j_bit_offset);
                }

                // --- 5.c: Check Clear Parameter P_clear[j][i] ---
                uint64_t clear_param_flat_idx = j * LAYER_BITSIZE + i; // Flattened index [row][col] within clear matrix
                // Add base offset to get index within the full current_param array
                uint64_t clear_param_word_idx = param_clear_base_offset_u64 + (clear_param_flat_idx / 64);
                uint64_t clear_param_bit_offset = clear_param_flat_idx % 64;

                // Check if the parameter bit allowing input 'i' to clear output 'j' is active
                bool clear_param_is_set = (current_param[clear_param_word_idx] >> clear_param_bit_offset) & 1ULL;

                if (clear_param_is_set) {
                    // AND the j-th bit's complement into the clear mask (i.e., set the j-th mask bit to 0)
                    clear_mask_u64[j_word_idx] &= ~(1ULL << j_bit_offset);
                }
            } // End loop output bit j
        } // End if input_bit_is_set
    } // End loop input bit i

    // --- Step 6: Apply the Clear Mask ---
    // Apply the accumulated clear mask to the output buffer
    for (size_t k = 0; k < LAYER_BITSIZE / 64; ++k) {
        output_layer_u64[k] &= clear_mask_u64[k];
    }

    // --- Step 7: Swap Pointers for Next Iteration ---
    // The calculation is complete. output_layer_u64 (originally td->layer2_u64)
    // now holds the new state. Swap the pointers so that for the *next* call
    // to layer_cal, this new state becomes layer1 (the input).
    uint64_t* tmp_u64 = td->layer1_u64;
    td->layer1_u64 = td->layer2_u64;
    td->layer2_u64 = tmp_u64;

    // Also swap the uint8_t pointers to match
    uint8_t* tmp_u8 = td->layer1_u8;
    td->layer1_u8 = td->layer2_u8;
    td->layer2_u8 = tmp_u8;

    // Now td->layer1 points to the result computed in this call.
    // td->layer2 points to the buffer used as input in this call (ready to be overwritten next time).
}


// --- Parameter Functions ---

// Initialize global best_param and thread-local random seeds
void param_init(uint64_t seed) {
    uint64_t current_rd = seed;
    printf("Initializing global best parameters with seed %llu...\n", (unsigned long long)seed);
    for (int i = 0; i < PARAM_U64_SIZE; i++) {
        current_rd = xorshift64(current_rd);
        best_param[i] = current_rd;
    }
     printf("Global parameters initialized.\n");

    // Initialize thread data structures
    printf("Initializing thread-local data...\n");
    for (int i = 0; i < THREAD_COUNT; ++i) {
        thread_data[i].id = i;
        // Assign layer pointers correctly within the allocated buffer
        thread_data[i].layer1_u64 = thread_data[i].layer_data;                             // Points to start
        thread_data[i].layer2_u64 = thread_data[i].layer_data + (LAYER_BITSIZE / 64);      // Points to midpoint
        thread_data[i].layer1_u8 = (uint8_t*)(thread_data[i].layer1_u64);
        thread_data[i].layer2_u8 = (uint8_t*)(thread_data[i].layer2_u64);
        // Give each thread a different starting random seed derived from the initial one
        current_rd = xorshift64(current_rd);
        thread_data[i].rd = current_rd ? current_rd : 1;  // Ensure rd is not 0
        // Initialize local param (can be done here or just before first mutation)
        // memset(thread_data[i].local_param, 0, sizeof(thread_data[i].local_param));
    }
    printf("Thread data initialized.\n");
}

// Mutates the thread's local_param based on the global best_param
void param_mutate(thread_data_t* td) {
    // Safely copy best_param to local_param before mutation
    pthread_mutex_lock(&best_param_mutex);
    memcpy(td->local_param, best_param, sizeof(best_param));
    pthread_mutex_unlock(&best_param_mutex);

    int mutations = 0;
    // Calculate threshold for mutation based on MUTATION_RATE
    // Avoid floating point inside the loop if possible
    uint64_t threshold = (uint64_t)(UINT64_MAX * MUTATION_RATE);
    if (threshold == 0 && MUTATION_RATE > 0.0) { // Handle very small rates
         threshold = 1;
    }


    for (int i = 0; i < PARAM_U64_SIZE; i++) {
        // Generate a random number for mutation check
        td->rd = xorshift64(td->rd);
        if (td->rd < threshold) {
            // Generate a new random value for the mutated parameter
            td->rd = xorshift64(td->rd);
            td->local_param[i] = td->rd;  // Mutate this element in the local copy
            mutations++;
        }
        // If not mutated, td->local_param[i] retains the value from best_param
    }
    // Optional: Print mutation count per thread for debugging
    // if (mutations > 0) {
    //     printf("Thread %d mutated %d params (threshold %llu)\n", td->id, mutations, (unsigned long long)threshold);
    // }
}

// --- Evaluation Function (Operates on Thread Data) ---

// Evaluates the given current_param using the thread's layer data
int64_t evaluate(thread_data_t* td, const uint64_t* current_param) {
    int64_t current_total_score = 0;
    uint8_t ch_out;
    uint8_t ch_in;
    uint8_t ch_correct;

    layer_reset(td);  // Reset layer state (IO and Memory) before evaluation

    if (train_size < 2)
        return INT64_MIN; // Cannot evaluate with less than 2 chars for a prediction pair

    // --- Phase 1: Process first half using direct input from train_data ---
    // Predict train_data[i+1] based on train_data[i]
    for (int64_t i = 0; i < train_size / 2; i++) {
        ch_in = train_data[i];
        ch_correct = train_data[i + 1];

        layer_setchar(td, ch_in);      // Set the input character (cleans IO, preserves memory)
        layer_cal(td, current_param);  // Calculate next state using the param set being evaluated
        // layer1 now holds the output state for this step

        current_total_score += layer_score(td, ch_correct); // Score the output against the correct next char

        // Optional: Early exit if prediction is wrong (can speed up finding good params initially)
        ch_out = layer_getchar(td);
        if (ch_out != ch_correct) {
            // Return a very low score or just the current score?
            // Returning current score allows differentiating slightly bad from very bad.
             return current_total_score;
        }
    }

    // --- Phase 2: Process second half using predicted input ---
    // Get the last predicted character from Phase 1 to start Phase 2.
    // If train_size was 2 or 3, Phase 1 ran once (i=0). layer_getchar() gives prediction for train_data[1].
    // If train_size was < 2, we already returned.
    if (train_size >= 2) {
         ch_in = layer_getchar(td); // Use the model's own prediction as the next input
    } else {
        // This part should not be reached due to the initial check, but for safety:
        return current_total_score; // Or INT64_MIN
    }


    // Predict train_data[i+1] based on the model's previous output ch_in (which was ch_out from previous step)
    for (int64_t i = train_size / 2; i < train_size - 1; i++) {
        ch_correct = train_data[i + 1];

        layer_setchar(td, ch_in);  // Use previous output character as input
        layer_cal(td, current_param);
        ch_out = layer_getchar(td); // Get the new prediction

        current_total_score += layer_score(td, ch_correct); // Score the prediction

        // Optional: Early exit
        if (ch_out != ch_correct) {
            return current_total_score;
        }
        ch_in = ch_out;  // Update input for the next iteration (autoregressive)
    }

    return current_total_score;
}

// --- Training ---

// Worker function for each thread
void* train_worker(void* arg) {
    thread_data_t* td = (thread_data_t*)arg;
    //int iterations_done_by_thread = 0; // Local counter if needed

    // Each thread pulls work until the global counter reaches TRAIN_COUNT
    while (1) {
        // Atomically get and increment the global iteration counter
        // __sync_fetch_and_add is a GCC/Clang atomic builtin
        int current_global_iter = __sync_fetch_and_add(&global_iterations, 1);

        if (current_global_iter >= TRAIN_COUNT) {
            break;  // All required iterations have been started/completed
        }

        param_mutate(td);                               // Mutate local param based on global best
        int64_t score = evaluate(td, td->local_param);  // Evaluate the local mutation

        // --- Critical Section: Check for improvement and update global best ---
        pthread_mutex_lock(&best_param_mutex);
        if (score > best_score) {
             // Found a new global best
            best_score = score;
            memcpy(best_param, td->local_param, sizeof(best_param)); // Copy the winning parameters

            // Optional: Unlock best_param mutex before printing to reduce contention
            // But keep print mutex locked for orderly output.
            pthread_mutex_unlock(&best_param_mutex); // Unlock early

            // Coordinated printing for new best score
            pthread_mutex_lock(&print_mutex);
            printf("Iter %d (Thread %d): New best score: %lld\n", current_global_iter + 1, td->id, score); // Use score directly
            pthread_mutex_unlock(&print_mutex);
        } else {
             // No improvement, just unlock the mutex
            pthread_mutex_unlock(&best_param_mutex);
        }
        // --- End Critical Section ---


        // Optional: Progress report less frequently
        // Print roughly 20 times during the training run
        const int report_interval = (TRAIN_COUNT / 20) > 0 ? (TRAIN_COUNT / 20) : 1;
        if ((current_global_iter + 1) % report_interval == 0) {
            pthread_mutex_lock(&print_mutex);
            // Read best_score again inside print lock for consistency, need best_param_mutex again
            int64_t current_best_for_print;
            pthread_mutex_lock(&best_param_mutex);
            current_best_for_print = best_score;
            pthread_mutex_unlock(&best_param_mutex);
            // Print status - using current_global_iter is fine here
            printf("--- Progress: Iter %d / %d --- Current best score: %lld\n", current_global_iter + 1, TRAIN_COUNT, current_best_for_print);
            pthread_mutex_unlock(&print_mutex);
        }

        // iterations_done_by_thread++; // Increment local counter if needed
    }

    // Optional: report how many iterations each thread did
    // printf("Thread %d finished, completed %d iterations.\n", td->id, iterations_done_by_thread);

    return NULL;
}


void train() {
    printf("Starting parallel training with %d threads...\n", THREAD_COUNT);
    printf("Total training iterations: %d\n", TRAIN_COUNT);

    // Calculate initial score (single threaded) using initial random best_param
    // Use thread_data[0] for this temporary calculation
    printf("Calculating initial score...\n");
    layer_reset(&thread_data[0]);                        // Reset state for thread 0
    best_score = evaluate(&thread_data[0], best_param);  // Evaluate initial random params
    printf("Initial best score: %lld\n", best_score);

    global_iterations = 0;  // Reset global iteration counter before starting threads

    pthread_t threads[THREAD_COUNT];

    // --- Create Threads ---
    printf("Creating %d worker threads...\n", THREAD_COUNT);
    for (int i = 0; i < THREAD_COUNT; i++) {
        if (pthread_create(&threads[i], NULL, train_worker, &thread_data[i]) != 0) {
            perror("Failed to create thread");
            // Attempt to clean up already created threads? Or just exit?
             // For simplicity, exit. In robust code, might try to join created threads.
            exit(EXIT_FAILURE);
        }
    }
    printf("Threads created.\n");

    // --- Wait for Threads to Finish ---
    printf("Waiting for threads to complete...\n");
    for (int i = 0; i < THREAD_COUNT; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("Failed to join thread");
            // Continue trying to join others? Or exit? Exit for simplicity.
            exit(EXIT_FAILURE);
        }
    }

    // --- Training Complete ---
    // Read final best_score (needs mutex for safety, though threads are done)
    pthread_mutex_lock(&best_param_mutex);
    int64_t final_best_score = best_score;
    pthread_mutex_unlock(&best_param_mutex);

    printf("Training finished. All threads joined.\n");
    printf("Final best score: %lld\n", final_best_score);
}


// --- Generation Function ---

// Uses the final best_param and thread_data[0]'s layers for generation
void generate(int64_t output_len) {
    if (output_len <= 0) {
        printf("Output length is zero or negative, skipping generation.\n");
        return;
    }
     printf("Generating text (%lld characters)...\n", (long long)output_len);

    // Use thread 0's data structure for generation.
    thread_data_t* td = &thread_data[0];
    layer_reset(td);                      // Reset state (IO and Memory) before generation

    uint8_t ch_current;

    // --- Priming Phase ---
    // Feed the input sequence to the model to set its internal state (memory)
    printf("Priming with input: ");
    fflush(stdout); // Ensure prompt prints before potential long pause
    for (int64_t i = 0; i < input_size; i++) {
        ch_current = input_data[i];
        // Print printable chars, '?' otherwise
        printf("%c", (ch_current > 31 && ch_current < 127) ? ch_current : '?');
        fflush(stdout);
        layer_setchar(td, ch_current);
        // Use the globally best parameters found during training
        layer_cal(td, best_param); // Pass best_param explicitly
    }
    printf("\n"); // Newline after priming input

    // --- Generation Phase ---
    // Get the first character to generate based on the state after priming.
    // The next character prediction is now in layer1 after the last layer_cal in the priming loop.
    ch_current = layer_getchar(td);

    printf("Generated output: ");
    fflush(stdout);

    // Buffer for generated output
    // Ensure output_data buffer is large enough or handle overflow
    if (output_len > TEXT_BITSIZE / 8) {
         fprintf(stderr, "\nWarning: Requested output length (%lld) exceeds buffer size (%d). Truncating to buffer size.\n", (long long)output_len, TEXT_BITSIZE / 8);
         output_len = TEXT_BITSIZE / 8;
    }

    int64_t generated_count = 0;
    for (int64_t i = 0; i < output_len; i++) {
        output_data[i] = ch_current; // Store the generated character
        generated_count++;

        // Print the character (printable or '?')
        printf("%c", (ch_current > 31 && ch_current < 127) ? ch_current : '?');
        fflush(stdout); // Print each char as it's generated

        // Prepare for the next step: Use the generated char as input
        layer_setchar(td, ch_current);
        layer_cal(td, best_param);       // Calculate next state using best parameters
        ch_current = layer_getchar(td);  // Get the next predicted character
    }
    printf("\n"); // Newline after generated output

    // --- Write Output File ---
    size_t written = file_write(output_data, generated_count, OUTPUT_PATH);
    if (written == (size_t)generated_count) { // Cast generated_count for comparison
        printf("Generated text successfully written to %s (%zu bytes)\n", OUTPUT_PATH, written);
    } else {
        fprintf(stderr, "Error writing generated text to %s\n", OUTPUT_PATH);
    }
}


// --- Main ---

int main() {
    printf("--- Simple Language Model (Parallel Training Version) ---\n");
    printf("Layer IO Size: %d bytes\n", LAYERIO_BYTESIZE);
    printf("Memory Size: %d bytes\n", MEMORY_BYTESIZE);
    printf("Total Layer Size: %d bytes (%d bits)\n", LAYER_BYTESIZE, LAYER_BITSIZE);
    printf("Parameter Size: %llu bits (%llu uint64_t, approx %.2f MB)\n",
           (unsigned long long)PARAM_BITSIZE,
           (unsigned long long)PARAM_U64_SIZE,
           (double)PARAM_BITSIZE / 8.0 / 1024.0 / 1024.0);
    printf("Threads: %d\n", THREAD_COUNT);
    printf("Train Iterations: %d\n", TRAIN_COUNT);
    printf("Mutation Rate: %.4f\n", MUTATION_RATE);
    printf("Random Seed: %d\n", RANDOM_SEED);
    printf("-------------------------------------------------------\n");


    printf("Reading training data from %s...\n", TRAIN_PATH);
    train_size = file_read(train_data, TRAIN_PATH);
    printf("Read %lld bytes of training data.\n", (long long)train_size);

    printf("Reading input data from %s...\n", INPUT_PATH);
    input_size = file_read(input_data, INPUT_PATH);
    printf("Read %lld bytes of input data.\n", (long long)input_size);

    if (train_size < 2) {
        fprintf(stderr, "Error: Training data requires at least 2 bytes for one input-output pair.\n");
        return EXIT_FAILURE;
    }

    // Initialize parameters, thread-local data, and random seeds
    param_init(RANDOM_SEED);

    // Initialize mutexes
    printf("Initializing mutexes...\n");
    if (pthread_mutex_init(&best_param_mutex, NULL) != 0) {
        perror("Mutex init failed (best_param_mutex)");
        return EXIT_FAILURE;
    }
    if (pthread_mutex_init(&print_mutex, NULL) != 0) {
        perror("Mutex init failed (print_mutex)");
        pthread_mutex_destroy(&best_param_mutex);  // Clean up previous mutex
        return EXIT_FAILURE;
    }
    printf("Mutexes initialized.\n");
    printf("-------------------------------------------------------\n");


    // Perform training using threads
    train();
    printf("-------------------------------------------------------\n");


    // Generate output using the best parameters found
    generate(OUTPUT_COUNT);
    printf("-------------------------------------------------------\n");


    // Clean up mutexes
    printf("Destroying mutexes...\n");
    pthread_mutex_destroy(&best_param_mutex);
    pthread_mutex_destroy(&print_mutex);

    printf("--- Program finished successfully ---\n");
    return EXIT_SUCCESS;
}