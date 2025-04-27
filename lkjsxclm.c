#include <errno.h>  // For perror
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>  // For sleep, usleep (optional)

// --- Configuration Constants ---
#define LAYERIO_BYTESIZE 256  // Size of the input/output part of a layer
#define MEMORY_BYTESIZE 256     // Size of the internal memory part of a layer
#define MUTATION_RATE 0.0001  // Probability of mutating a parameter byte (AFTER GRPO)
#define PRUNING_LIMIT 2       // Max wrong predictions before stopping evaluation
#define THINK_LIMIT 3         // Max consecutive THINK tokens allowed
#define THINK_REWARD 16        // Score bonus for emitting THINK_TOKEN (up to THINK_LIMIT times)

#define THREAD_COUNT 14      // Number of parallel evaluation threads
#define TRAIN_COUNT 10000  // Number of evaluation cycles per thread
#define OUTPUT_COUNT 1000    // Number of characters to generate in the output
#define RANDOM_SEED 123      // Seed for deterministic randomness (0 for time-based)

#define TEXT_BYTESIZE (1024 * 1024 * 16)  // Max size for text buffers
#define TEXT_SETSIZE 1024                 // Max number of lines expected in training data
#define TRAIN_PATH "./train.txt"          // Path to training data file
#define INPUT_PATH "./input.txt"          // Path to input data file for generation
#define OUTPUT_PATH "./output.txt"        // Path to output file for generated text

// --- GRPO Configuration ---
#define GRPO_GROUP_SIZE 8       // Number of elite policies to maintain (must be >= 2)
#define GRPO_SCALE_SHIFT 4      // Right shift for scaling difference vector (e.g., 4 => divide by 16)

// --- Derived Constants ---
#define THINK_TOKEN '\t'
#define LAYER_BYTESIZE (LAYERIO_BYTESIZE + MEMORY_BYTESIZE)          // Total size of a layer's state
#define PARAM_BYTESIZE (LAYER_BYTESIZE * (LAYER_BYTESIZE + 1))       // Size of parameters (Bias + Weights)
#define THREAD_TRAIN_COUNT (TRAIN_COUNT / THREAD_COUNT)              // Cycles per thread
#define LAYER_CAL_SCALE_SHIFT 11                                     // Right shift amount in layer_cal
#define SCORE_PENALTY_PER_ACTIVATION 1                               // Penalty for each unit *below* max activation
#define SCORE_REWARD_CORRECT_ACTIVATION (LAYERIO_BYTESIZE / 32)      // Reward multiplier for activation at correct index
#define SCORE_BONUS_EXACT_MATCH (LAYERIO_BYTESIZE * UINT8_MAX / 32)  // Bonus for correct prediction

#if GRPO_GROUP_SIZE < 2
#error "GRPO_GROUP_SIZE must be at least 2"
#endif

// --- Type Definitions ---

// Simple string view structure
typedef struct {
    const uint8_t* data;
    size_t size;
} string_t;

// Structure to hold an elite policy and its score
typedef struct {
    uint8_t params[PARAM_BYTESIZE]; // Parameters stored as raw bytes
    int64_t score;
} elite_member_t;

// Thread-specific data structure
typedef struct {
    int64_t tid;         // Thread ID
    uint8_t* layer1_u8;  // Pointer to layer 1 data (current input/state)
    uint8_t* layer2_u8;  // Pointer to layer 2 data (next state buffer)
    int8_t* param_i8;    // Pointer to parameters (as int8_t)
    uint64_t rd;         // Thread-local random state (for xorshift64)

    // Buffers allocated per thread
    uint8_t layer_data[LAYER_BYTESIZE * 2]; // Double buffer for layer states
    uint8_t param_bytes[PARAM_BYTESIZE];    // Local copy of parameters for mutation/evaluation

    // Temporary buffer for GRPO parent selection (to avoid reading elite group while holding lock for too long)
    uint8_t grpo_parent1[PARAM_BYTESIZE];
    uint8_t grpo_parent2[PARAM_BYTESIZE];

} thread_t;

// Global shared data structure
typedef struct {
    int64_t best_score;      // Best evaluation score found so far (overall)
    size_t train_size;       // Actual size of training data read
    size_t input_size;       // Actual size of input data read
    size_t train_set_count;  // Number of line pairs in training data

    // Data buffers
    uint8_t train_data[TEXT_BYTESIZE];
    uint8_t input_data[TEXT_BYTESIZE];
    uint8_t output_data[TEXT_BYTESIZE];  // Buffer for generated output

    // Array of string views for training lines (pairs: prompt, response)
    string_t train_set[TEXT_SETSIZE * 2];

    // Best parameters found globally (single best)
    uint8_t best_param[PARAM_BYTESIZE];

    // --- GRPO Elite Group ---
    elite_member_t elite_group[GRPO_GROUP_SIZE];
    int elite_group_count; // Number of valid members currently in the group

    // Thread management
    pthread_mutex_t global_update_mutex; // Mutex protecting elite_group, elite_group_count, best_score, best_param

} global_t;

// --- Global Variables ---
global_t global_data;
thread_t thread_data[THREAD_COUNT];
volatile bool keep_running = true; // Flag to signal threads to stop (optional)

// --- Function Prototypes ---
uint64_t xorshift64(uint64_t x);
size_t file_read(uint8_t* dst, const char* filename, size_t max_size);
size_t file_write(const uint8_t* src, size_t size, const char* filename);
void initialize_parameters(uint8_t* params, size_t size, uint64_t* seed);
void initialize_elite_group(uint64_t* seed);
void update_elite_group(const uint8_t* candidate_params, int64_t candidate_score); // Must be called with lock held
void param_update_grpo(thread_t* td);
void layer_reset(thread_t* td);
void layer_setchar(thread_t* td, uint8_t index);
uint8_t layer_getchar(thread_t* td);
int64_t layer_score(thread_t* td, uint8_t predicted_char, uint8_t correct_char);
void layer_cal(thread_t* td);
int64_t evaluate(thread_t* td);
void* thread_func(void* arg);
void generate_output(const char* output_filename);
int parse_train_data();

// --- Random Number Generation ---
uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

// --- File I/O --- (Identical to original, omitted for brevity)
size_t file_read(uint8_t* dst, const char* filename, size_t max_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open file '%s': ", filename);
        perror(NULL);
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) { /* ... error handling ... */ fclose(fp); return 0; }
    long file_size_long = ftell(fp);
    if (file_size_long < 0) { /* ... error handling ... */ fclose(fp); return 0; }
    size_t file_size = (size_t)file_size_long;
    if (file_size >= max_size) {
        fprintf(stderr, "WARNING: File '%s' (size %zu) exceeds buffer size (%zu). Truncating.\n", filename, file_size, max_size);
        file_size = max_size - 1;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) { /* ... error handling ... */ fclose(fp); return 0; }
    size_t read_count = fread(dst, 1, file_size, fp);
    if (read_count != file_size) { /* ... error handling ... */ }
    fclose(fp);
    return read_count;
}

size_t file_write(const uint8_t* src, size_t size, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { /* ... error handling ... */ return 0; }
    size_t written_count = fwrite(src, 1, size, fp);
    if (written_count != size) { /* ... error handling ... */ }
    if (fclose(fp) != 0) { /* ... error handling ... */ }
    return written_count;
}


// --- Parameter Initialization ---
void initialize_parameters(uint8_t* params, size_t size, uint64_t* seed) {
    printf("Initializing %zu parameter bytes with random values...\n", size);
    for (size_t i = 0; i < size; ++i) {
        *seed = xorshift64(*seed);
        // Treat bytes as signed int8_t for parameters
        params[i] = (int8_t)((*seed) & 0xFF);
    }
}

// --- GRPO Elite Group Management ---

// Initialize the elite group (call before starting threads)
void initialize_elite_group(uint64_t* seed) {
    printf("Initializing GRPO elite group (size %d)...\n", GRPO_GROUP_SIZE);
    global_data.elite_group_count = 0;
    // Initialize scores to minimum possible
    for (int i = 0; i < GRPO_GROUP_SIZE; ++i) {
        global_data.elite_group[i].score = LLONG_MIN;
    }

    // Seed the first elite member with the initial best parameters
    if (GRPO_GROUP_SIZE > 0) {
        memcpy(global_data.elite_group[0].params, global_data.best_param, PARAM_BYTESIZE);
        // Evaluate the initial random parameters to give the first member a score?
        // This requires a temporary thread context or careful setup.
        // Simpler: Assign the initial best_score (LLONG_MIN) or just leave it.
        // Let's assign the initial best_score. Evaluation will quickly replace it.
        global_data.elite_group[0].score = global_data.best_score; // Initially LLONG_MIN
        global_data.elite_group_count = 1;
        printf("Seeded elite group member 0 with initial parameters.\n");
    }
     // Optionally, fill more slots with random mutations if desired
     /*
     for (int i = 1; i < GRPO_GROUP_SIZE; ++i) {
         initialize_parameters(global_data.elite_group[i].params, PARAM_BYTESIZE, seed);
         global_data.elite_group[i].score = LLONG_MIN; // Needs evaluation later
         global_data.elite_group_count++;
     }
     */
}

// Update the elite group if the candidate is good enough.
// MUST be called with global_update_mutex HELD.
void update_elite_group(const uint8_t* candidate_params, int64_t candidate_score) {
    int worst_elite_idx = -1;
    int64_t worst_elite_score = LLONG_MAX;

    // Check if candidate is better than the overall best
    bool new_best_overall = false;
    if (candidate_score > global_data.best_score) {
        global_data.best_score = candidate_score;
        memcpy(global_data.best_param, candidate_params, PARAM_BYTESIZE);
        new_best_overall = true;
        // Note: Printing should ideally happen outside the lock or be very fast.
        // printf(">>> New global best score: %lld\n", candidate_score); // Moved print outside
    }

    // Find the index of the worst member in the elite group
    if (global_data.elite_group_count == GRPO_GROUP_SIZE) {
        for (int i = 0; i < GRPO_GROUP_SIZE; ++i) {
            if (global_data.elite_group[i].score < worst_elite_score) {
                worst_elite_score = global_data.elite_group[i].score;
                worst_elite_idx = i;
            }
        }
    } else {
        // Group is not full, the "worst" is effectively the next empty slot
        worst_elite_idx = global_data.elite_group_count;
        worst_elite_score = LLONG_MIN; // Any score is better than an empty slot's implicit score
    }

    // If the candidate is better than the worst elite (or group is not full)
    if (candidate_score > worst_elite_score) {
        // Replace the worst member (or fill the next slot)
        if (worst_elite_idx != -1) { // Should always be true based on logic above
             // Avoid duplicate entries if the candidate is already the best overall and in the group
            bool already_in_group = false;
            if (new_best_overall) { // Only need check if it's the new best
                 for(int i=0; i < global_data.elite_group_count; ++i) {
                     // Simple check: are parameters identical to an existing elite member?
                     // (Expensive check, maybe skip or use score as proxy?)
                     // Let's skip the deep comparison for performance under lock.
                     // If score matches, assume it might be a duplicate (not perfect)
                     if (global_data.elite_group[i].score == candidate_score) {
                         // Check if it's the *same* memory location being added back? No, candidate_params is thread local.
                         // Let's allow potential score duplicates for now.
                         // already_in_group = (memcmp(global_data.elite_group[i].params, candidate_params, PARAM_BYTESIZE) == 0);
                         // if (already_in_group) break;
                     }
                 }
            }

            if (!already_in_group) {
                memcpy(global_data.elite_group[worst_elite_idx].params, candidate_params, PARAM_BYTESIZE);
                global_data.elite_group[worst_elite_idx].score = candidate_score;
                // Increment count if we filled an empty slot
                if (worst_elite_idx == global_data.elite_group_count && global_data.elite_group_count < GRPO_GROUP_SIZE) {
                    global_data.elite_group_count++;
                     // printf("Added new elite member %d, score %lld\n", worst_elite_idx, candidate_score);
                } else {
                    // printf("Replaced elite member %d, score %lld\n", worst_elite_idx, candidate_score);
                }
            }
        }
    }
}


// --- Evolutionary Algorithm Components ---

// Update thread-local parameters using GRPO and mutation
void param_update_grpo(thread_t* td) {
    int parent1_idx = -1;
    int parent2_idx = -1;
    int current_elite_count = 0;
    bool use_grpo = false;

    // --- Parent Selection (Read-only access to elite group) ---
    pthread_mutex_lock(&global_data.global_update_mutex);
    current_elite_count = global_data.elite_group_count;
    if (current_elite_count >= 2) {
        // Randomly select two distinct parents from the elite group
        td->rd = xorshift64(td->rd);
        parent1_idx = td->rd % current_elite_count;
        td->rd = xorshift64(td->rd);
        parent2_idx = td->rd % current_elite_count;
        // Ensure parents are distinct
        if (current_elite_count > 1) { // Avoid infinite loop if count is 1 (shouldn't happen with >=2 check)
           while (parent1_idx == parent2_idx) {
               td->rd = xorshift64(td->rd);
               parent2_idx = td->rd % current_elite_count;
           }
        }

        // Copy parent parameters to thread-local buffers to minimize time holding lock
        memcpy(td->grpo_parent1, global_data.elite_group[parent1_idx].params, PARAM_BYTESIZE);
        memcpy(td->grpo_parent2, global_data.elite_group[parent2_idx].params, PARAM_BYTESIZE);
        use_grpo = true;
    } else {
        // Fallback: Copy the single best parameter if elite group is too small
        memcpy(td->param_bytes, global_data.best_param, PARAM_BYTESIZE);
    }
    pthread_mutex_unlock(&global_data.global_update_mutex);


    // --- GRPO Update (if parents were selected) ---
    if (use_grpo) {
        // Use parent1 as the base for the new parameters
        memcpy(td->param_bytes, td->grpo_parent1, PARAM_BYTESIZE);

        // Cast pointers once for easier access
        int8_t* base_params = (int8_t*)td->param_bytes;
        const int8_t* p1_params = (const int8_t*)td->grpo_parent1; // Base is already p1
        const int8_t* p2_params = (const int8_t*)td->grpo_parent2;

        for (size_t i = 0; i < PARAM_BYTESIZE; i++) {
            // Calculate difference using int16_t to avoid intermediate overflow/underflow
            int16_t p1_val = (int16_t)p1_params[i];
            int16_t p2_val = (int16_t)p2_params[i];
            int16_t diff = p2_val - p1_val;

            // Scale the difference using integer right shift
            // Shift applied to signed value performs arithmetic shift (preserves sign)
            int16_t scaled_diff = diff >> GRPO_SCALE_SHIFT;

            // Add scaled difference to the base parameter (which is p1)
            int16_t updated_val = (int16_t)base_params[i] + scaled_diff; // Base params already in td->param_bytes

            // Clamp the result back to the int8_t range [-128, 127]
            if (updated_val > INT8_MAX) {
                updated_val = INT8_MAX;
            } else if (updated_val < INT8_MIN) {
                updated_val = INT8_MIN;
            }

            // Store the updated parameter
            base_params[i] = (int8_t)updated_val;
        }
    }
    // If not use_grpo, td->param_bytes already contains the best_param from the fallback case.

    // --- Standard Mutation (applied AFTER GRPO or fallback copy) ---
    const uint64_t mutation_threshold = (uint64_t)(MUTATION_RATE * UINT64_MAX);
    int8_t* current_params = (int8_t*)td->param_bytes; // Use signed pointer

    for (size_t i = 0; i < PARAM_BYTESIZE; i++) {
        td->rd = xorshift64(td->rd);
        if (td->rd < mutation_threshold) {
            // Mutate this byte to a new random int8_t value
            td->rd = xorshift64(td->rd);
            current_params[i] = (int8_t)(td->rd & 0xFF); // Generate random byte and cast to signed
        }
    }
}


// --- Layer Computation --- (Identical to original, omitted for brevity)
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
        // fprintf(stderr, "Warning: layer_setchar index %u out of bounds (%d), setting index 0\n", index, LAYERIO_BYTESIZE);
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
    // Penalty scaled down to avoid dominance, also ensures non-negative intermediate values
    for (size_t i = 0; i < LAYERIO_BYTESIZE; i++) {
        score -= (int64_t)(UINT8_MAX - td->layer1_u8[i]) * SCORE_PENALTY_PER_ACTIVATION;
    }
     // Reward activation at the correct index
    score += (int64_t)td->layer1_u8[correct_char] * SCORE_REWARD_CORRECT_ACTIVATION;
    // Bonus if the predicted character (max activation index) is correct
    if (predicted_char == correct_char) {
        score += SCORE_BONUS_EXACT_MATCH;
    }
    return score; // Score can be negative now due to penalty
}

void layer_cal(thread_t* td) {
    const int8_t* params = td->param_i8;
    int param_idx = 0;
    for (size_t dst_i = 0; dst_i < LAYER_BYTESIZE; dst_i++) {
        int64_t sum = params[param_idx++]; // Bias
        for (size_t src_i = 0; src_i < LAYER_BYTESIZE; src_i++) {
             // Ensure non-negative input for multiplication if needed, but layer1 is uint8_t
             // Multiplication: uint8 * int8 -> potential negative. Use int64_t accumulator.
            sum += (int64_t)td->layer1_u8[src_i] * (int64_t)params[param_idx++];
        }
        // Clamp negative values *before* scaling shift
        if (sum < 0) sum = 0;
        // Scale down using right shift
        sum >>= LAYER_CAL_SCALE_SHIFT;
        // Clamp to uint8_t max
        if (sum > UINT8_MAX) sum = UINT8_MAX;
        td->layer2_u8[dst_i] = (uint8_t)sum;
    }
    uint8_t* tmp_u8 = td->layer1_u8;
    td->layer1_u8 = td->layer2_u8;
    td->layer2_u8 = tmp_u8;
}


// --- Evaluation Function --- (Identical to original, omitted for brevity)
int64_t evaluate(thread_t* td) {
    int64_t total_score = 0;
    // Iterate through training data line pairs (prompt, response)
    for (size_t pair_idx = 0; pair_idx < global_data.train_set_count; ++pair_idx) {
        const string_t* prompt_line = &global_data.train_set[pair_idx * 2];
        const string_t* response_line = &global_data.train_set[pair_idx * 2 + 1];
        layer_reset(td);
        int64_t current_wrong_streak = 0;
        int64_t current_think_streak = 0;
        // 1. Prime with prompt
        for (size_t i = 0; i < prompt_line->size; ++i) {
            layer_setchar(td, prompt_line->data[i]);
            layer_cal(td);
        }
        // 2. Evaluate on response
        for (size_t i = 0; i < response_line->size - 1; ++i) {
            uint8_t ch_in = response_line->data[i];
            uint8_t ch_correct = response_line->data[i + 1];
            layer_setchar(td, ch_in);
            layer_cal(td);
            uint8_t ch_predicted = layer_getchar(td);
            int64_t step_score = layer_score(td, ch_predicted, ch_correct);
            total_score += step_score;

            if (ch_predicted == ch_correct) {
                current_wrong_streak = 0;
                current_think_streak = 0;
            } else if (ch_predicted == THINK_TOKEN) {
                current_think_streak++;
                if (current_think_streak <= THINK_LIMIT) {
                    total_score += THINK_REWARD;
                    i--; // Re-evaluate same input
                    continue;
                } else {
                    current_wrong_streak++;
                    current_think_streak = 0;
                }
            } else {
                current_wrong_streak++;
                current_think_streak = 0;
            }
            if (current_wrong_streak > PRUNING_LIMIT) {
                break;
            }
        }
    }
    return total_score;
}

// --- Threading ---

// Function executed by each worker thread
void* thread_func(void* arg) {
    thread_t* td = (thread_t*)arg;
    int64_t local_best_score = LLONG_MIN; // Track local best for less frequent global checks/prints

    for (size_t train_step = 0; train_step < THREAD_TRAIN_COUNT; ++train_step) {
        // 1. Generate new parameters using GRPO + mutation
        param_update_grpo(td);

        // 2. Evaluate the parameters
        int64_t current_score = evaluate(td);

        // 3. Try to update the global elite group and best score (thread-safe)
        // Only lock if the score is potentially good enough to avoid unnecessary contention.
        // Check against local best first, then global best (read without lock is slightly racy but ok for check)
        if (current_score > local_best_score) {
             local_best_score = current_score; // Update local best

             // Check if potentially better than global best or worth adding to elite group
             // Reading best_score without lock is a race condition, but only used for filtering.
             // Worst case: we lock unnecessarily. Better than always locking.
             // A more robust check would involve comparing to the worst elite score, but that requires reading the group.
             // Let's just check against the global best score for simplicity here.
             if (current_score > global_data.best_score) {
                int64_t global_best_before_lock = LLONG_MIN; // To check if update happened

                pthread_mutex_lock(&global_data.global_update_mutex);
                global_best_before_lock = global_data.best_score; // Get accurate value under lock
                update_elite_group(td->param_bytes, current_score); // Handles elite and global best update
                bool updated_global = (global_data.best_score > global_best_before_lock);
                pthread_mutex_unlock(&global_data.global_update_mutex);

                // Print outside the lock if a new global best was confirmed
                 if (updated_global) {
                    // Print less frequently to avoid spamming console
                    // if (train_step % (THREAD_TRAIN_COUNT / 20) == 0 || current_score > global_best_before_lock)
                    {
                         printf("T:%2lld, Step:%zu(%d%%), New best: %lld (EliteCnt:%d)\n",
                               td->tid, train_step, (int)((100 * train_step) / THREAD_TRAIN_COUNT),
                               current_score, global_data.elite_group_count); // Read count outside lock is slightly racy but ok for info
                    }
                 }
             }
        }
    }
    return NULL;
}

// --- Output Generation --- (Identical to original, uses global_data.best_param)
void generate_output(const char* output_filename) {
    printf("\n--- Generating Output ---\n");
    printf("Using best parameters found during training (Score: %lld).\n", global_data.best_score);
    thread_t* td = &thread_data[0]; // Re-purpose thread 0's buffers
    memcpy(td->param_bytes, global_data.best_param, PARAM_BYTESIZE); // Load ABSOLUTE best
    td->param_i8 = (int8_t*)td->param_bytes;
    layer_reset(td);
    uint8_t current_char;
    printf("Priming model with input sequence (%zu bytes from %s)...\n", global_data.input_size, INPUT_PATH);
    if (global_data.input_size == 0) {
        current_char = 0;
    } else {
        for (size_t i = 0; i < global_data.input_size; ++i) {
            current_char = global_data.input_data[i];
            layer_setchar(td, current_char);
            layer_cal(td);
        }
        current_char = layer_getchar(td); // First predicted char
    }
    printf("Generating %d characters...\n", OUTPUT_COUNT);
    size_t output_idx = 0;
    int think_streak = 0;
    while (output_idx < OUTPUT_COUNT) {
        if (current_char != THINK_TOKEN) {
            global_data.output_data[output_idx++] = current_char;
            think_streak = 0;
        } else {
            think_streak++;
             // Optional: Limit consecutive thinks in generation?
             // printf("Warning: Generated THINK_TOKEN (streak %d)\n", think_streak);
        }
        layer_setchar(td, current_char); // Use generated char (or THINK) as next input
        layer_cal(td);
        current_char = layer_getchar(td); // Predict next char
    }
    printf("Writing %zu generated characters to %s...\n", output_idx, output_filename);
    size_t written = file_write(global_data.output_data, output_idx, output_filename);
    if (written == output_idx) {
        printf("Successfully wrote %zu characters.\n", written);
    } else {
        fprintf(stderr, "ERROR: Failed to write the full generated output to %s (wrote %zu).\n", output_filename, written);
    }
}

// --- Training Data Parsing --- (Identical to original, omitted for brevity)
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
        if (current >= end) break;
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


// --- Main Function ---
int main() {
    printf("--- Language Model Training & Generation (C + GRPO) ---\n");
    printf("Configuration:\n");
    printf("  Layer IO Size: %d, Memory Size: %d, Total Layer: %d\n", LAYERIO_BYTESIZE, MEMORY_BYTESIZE, LAYER_BYTESIZE);
    printf("  Param Size: %zu bytes\n", (size_t)PARAM_BYTESIZE); // Use %zu for size_t
    printf("  Mutation Rate: %f, Pruning Limit: %d\n", MUTATION_RATE, PRUNING_LIMIT);
    printf("  Threads: %d, Train Cycles (Total): %d, Cycles/Thread: %d\n", THREAD_COUNT, TRAIN_COUNT, THREAD_TRAIN_COUNT);
    printf("  Output Length: %d, Random Seed: %d\n", OUTPUT_COUNT, RANDOM_SEED);
    printf("  GRPO Group Size: %d, GRPO Scale Shift: %d\n", GRPO_GROUP_SIZE, GRPO_SCALE_SHIFT);
    printf("------------------------------------------------\n");

    // --- Initialization ---
    printf("Initializing...\n");
    global_data.best_score = LLONG_MIN;
    global_data.train_size = 0;
    global_data.input_size = 0;
    global_data.train_set_count = 0;
    memset(global_data.best_param, 0, PARAM_BYTESIZE);

    uint64_t master_seed = (RANDOM_SEED == 0) ? (uint64_t)time(NULL) : (uint64_t)RANDOM_SEED;
    master_seed = xorshift64(master_seed);
    if (master_seed == 0) master_seed = 1;

    // Initialize best parameters randomly (will also seed elite group)
    initialize_parameters(global_data.best_param, PARAM_BYTESIZE, &master_seed);

    // Initialize mutex
    if (pthread_mutex_init(&global_data.global_update_mutex, NULL) != 0) {
        perror("Mutex initialization failed");
        return EXIT_FAILURE;
    }

    // Initialize the GRPO elite group (must happen *after* best_param init and *before* threads start)
    initialize_elite_group(&master_seed);

    // Load data
    printf("Loading training data from %s...\n", TRAIN_PATH);
    global_data.train_size = file_read(global_data.train_data, TRAIN_PATH, TEXT_BYTESIZE);
    if (global_data.train_size == 0) { /* ... error handling ... */ pthread_mutex_destroy(&global_data.global_update_mutex); return EXIT_FAILURE; }
    printf("Loaded %zu bytes of training data.\n", global_data.train_size);

    if (parse_train_data() != 0) { /* ... error handling ... */ pthread_mutex_destroy(&global_data.global_update_mutex); return EXIT_FAILURE; }

    printf("Loading input data from %s...\n", INPUT_PATH);
    global_data.input_size = file_read(global_data.input_data, INPUT_PATH, TEXT_BYTESIZE);
    printf("Loaded %zu bytes of input data.\n", global_data.input_size);

    // Initialize thread-specific data
    printf("Initializing %d threads...\n", THREAD_COUNT);
    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        thread_data[i].tid = i;
        master_seed = xorshift64(master_seed);
        thread_data[i].rd = (master_seed == 0) ? 1 : master_seed;
        thread_data[i].layer1_u8 = thread_data[i].layer_data;
        thread_data[i].layer2_u8 = thread_data[i].layer_data + LAYER_BYTESIZE;
        thread_data[i].param_i8 = (int8_t*)thread_data[i].param_bytes; // Point to local param buffer
        // param_bytes will be filled by param_update_grpo at the start of each loop
    }

    // --- Training Phase ---
    printf("\n--- Starting Training Phase (GRPO) ---\n");
    pthread_t threads[THREAD_COUNT];
    time_t start_time = time(NULL);

    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        if (pthread_create(&threads[i], NULL, thread_func, &thread_data[i]) != 0) {
            perror("Failed to create thread");
            keep_running = false;
            pthread_mutex_destroy(&global_data.global_update_mutex);
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
    printf("Final elite group count: %d\n", global_data.elite_group_count);
    // Optionally print final elite scores
    printf("Elite scores: [ ");
    for(int i=0; i<global_data.elite_group_count; ++i) printf("%lld ", global_data.elite_group[i].score);
    printf("]\n");


    // Clean up mutex
    pthread_mutex_destroy(&global_data.global_update_mutex);

    // --- Output Generation Phase ---
    generate_output(OUTPUT_PATH);

    printf("\n--- Program Finished ---\n");
    return EXIT_SUCCESS;
}