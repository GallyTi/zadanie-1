#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

#define X_SIZE 1024
#define Y_SIZE 1024
#define Z_SIZE 314

#define TOTAL_VOXELS ((size_t)X_SIZE * Y_SIZE * Z_SIZE)
#define THRESHOLD 25
#define BITS_PER_COORD 10

typedef struct {
    int thread_id;
    size_t start_idx;
    size_t end_idx;
    uint8_t *data;
    uint32_t *morton_codes;
    size_t code_count;
    size_t max_codes;
} thread_data_t;

// Function to compare uint32_t values for qsort
int compare_uint32(const void *a, const void *b) {
    uint32_t arg1 = *(const uint32_t *)a;
    uint32_t arg2 = *(const uint32_t *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// Function to interleave bits for Morton code
uint32_t expand_bits(uint32_t x) {
    x &= 0x3FF; // Ensure x is 10 bits
    x = (x | (x << 16)) & 0x30000FF;
    x = (x | (x << 8))  & 0x300F00F;
    x = (x | (x << 4))  & 0x30C30C3;
    x = (x | (x << 2))  & 0x9249249;
    return x;
}

// Function to compute Morton code from x, y, z
uint32_t morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

// Thread function
void *thread_function(void *arg) {
    thread_data_t *thread_data = (thread_data_t *)arg;

    size_t start = thread_data->start_idx;
    size_t end = thread_data->end_idx;
    uint8_t *data = thread_data->data;
    size_t max_codes = thread_data->max_codes;

    uint32_t *morton_codes = malloc(max_codes * sizeof(uint32_t));
    if (!morton_codes) {
        fprintf(stderr, "Thread %d: Failed to allocate morton_codes array\n", thread_data->thread_id);
        pthread_exit(NULL);
    }

    size_t code_count = 0;

    for (size_t i = start; i < end; ++i) {
        uint8_t value = data[i];
        if (value > THRESHOLD) {
            // Compute x, y, z from index i
            size_t idx = i;
            uint32_t x = idx % X_SIZE;
            idx /= X_SIZE;
            uint32_t y = idx % Y_SIZE;
            uint32_t z = idx / Y_SIZE;

            uint32_t morton_code = morton_encode(x, y, z);

            // Check if morton_codes[] needs to be reallocated
            if (code_count >= max_codes) {
                size_t new_size = max_codes * 2;
                uint32_t *new_array = realloc(morton_codes, new_size * sizeof(uint32_t));
                if (!new_array) {
                    fprintf(stderr, "Thread %d: Failed to reallocate morton_codes array\n", thread_data->thread_id);
                    free(morton_codes);
                    pthread_exit(NULL);
                }
                morton_codes = new_array;
                max_codes = new_size;
            }
            morton_codes[code_count++] = morton_code;
        }
    }

    thread_data->morton_codes = morton_codes;
    thread_data->code_count = code_count;

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s num_threads\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[1]);
    if (num_threads <= 0) {
        printf("Invalid number of threads\n");
        return 1;
    }

    // Allocate data[]
    uint8_t *data = malloc(TOTAL_VOXELS * sizeof(uint8_t));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate data array\n");
        return 1;
    }

    // Read c8.raw
    FILE *fp = fopen("c8.raw", "rb");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open c8.raw\n");
        free(data);
        return 1;
    }
    size_t items_read = fread(data, sizeof(uint8_t), TOTAL_VOXELS, fp);
    fclose(fp);
    if (items_read != TOTAL_VOXELS) {
        fprintf(stderr, "Error: Failed to read data from c8.raw\n");
        free(data);
        return 1;
    }

    // Timing starts here
    clock_t start_time = clock();

    // Create threads
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));
    if (!threads || !thread_data) {
        fprintf(stderr, "Error: Failed to allocate threads or thread_data\n");
        free(data);
        return 1;
    }

    size_t voxels_per_thread = TOTAL_VOXELS / num_threads;
    size_t remainder = TOTAL_VOXELS % num_threads;

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].start_idx = i * voxels_per_thread;
        thread_data[i].end_idx = thread_data[i].start_idx + voxels_per_thread;
        if (i == num_threads - 1) {
            thread_data[i].end_idx += remainder;
        }
        thread_data[i].data = data;
        thread_data[i].max_codes = 1000000; // Initial size, will expand as needed
        thread_data[i].morton_codes = NULL;
        thread_data[i].code_count = 0;

        int rc = pthread_create(&threads[i], NULL, thread_function, (void *)&thread_data[i]);
        if (rc) {
            fprintf(stderr, "Error creating thread %d\n", i);
            free(data);
            free(threads);
            free(thread_data);
            return 1;
        }
    }

    // Wait for threads to finish
    size_t total_active_voxels = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        total_active_voxels += thread_data[i].code_count;
    }

    // Combine morton_codes from threads
    uint32_t *voxels = malloc(total_active_voxels * sizeof(uint32_t));
    if (!voxels) {
        fprintf(stderr, "Error: Failed to allocate voxels array\n");
        free(data);
        free(threads);
        free(thread_data);
        return 1;
    }
    size_t offset = 0;
    for (int i = 0; i < num_threads; ++i) {
        memcpy(voxels + offset, thread_data[i].morton_codes, thread_data[i].code_count * sizeof(uint32_t));
        offset += thread_data[i].code_count;
        free(thread_data[i].morton_codes);
    }

    // Sort voxels[]
    qsort(voxels, total_active_voxels, sizeof(uint32_t), compare_uint32);

    // Timing ends here
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Output number of active voxels
    printf("Number of active voxels: %zu\n", total_active_voxels);

    // Output first 10 Morton codes
    printf("First 10 Morton codes:\n");
    for (size_t i = 0; i < 10 && i < total_active_voxels; ++i) {
        printf("%u\n", voxels[i]);
    }

    // Verify if voxels[] is sorted
    int is_sorted = 1;
    for (size_t i = 1; i < total_active_voxels; ++i) {
        if (voxels[i - 1] > voxels[i]) {
            is_sorted = 0;
            fprintf(stderr, "Array is not sorted at index %zu\n", i);
            break;
        }
    }
    if (is_sorted) {
        printf("Morton codes are correctly sorted.\n");
    } else {
        printf("Morton codes are NOT correctly sorted.\n");
    }

    // Save Morton codes to file
    FILE *out_fp = fopen("morton_codes_pthread.txt", "w");
    if (out_fp) {
        for (size_t i = 0; i < total_active_voxels; ++i) {
            fprintf(out_fp, "%u\n", voxels[i]);
        }
        fclose(out_fp);
        printf("Morton codes saved to morton_codes_pthread.txt\n");
    } else {
        fprintf(stderr, "Error: Failed to open output file for writing.\n");
    }

    // Output processing time
    printf("Processing time with %d threads: %f seconds\n", num_threads, total_time);

    // Free resources
    free(voxels);
    free(data);
    free(threads);
    free(thread_data);

    return 0;
}
