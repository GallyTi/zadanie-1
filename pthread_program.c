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
    size_t *code_count;
} thread_data_t;

int compare_uint32(const void *a, const void *b) {
    uint32_t arg1 = *(const uint32_t *)a;
    uint32_t arg2 = *(const uint32_t *)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

uint32_t expand_bits(uint32_t x) {
    x &= 0x3FF;
    x = (x | (x << 16)) & 0x30000FF;
    x = (x | (x << 8)) & 0x300F00F;
    x = (x | (x << 4)) & 0x30C30C3;
    x = (x | (x << 2)) & 0x9249249;
    return x;
}

uint32_t morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

void *thread_function(void *arg) {
    thread_data_t *thread_data = (thread_data_t *)arg;
    uint8_t *data = thread_data->data;
    size_t start = thread_data->start_idx;
    size_t end = thread_data->end_idx;

    size_t code_count = 0;
    for (size_t i = start; i < end; ++i) {
        if (data[i] > THRESHOLD) {
            size_t idx = i;
            uint32_t x = idx % X_SIZE;
            idx /= X_SIZE;
            uint32_t y = idx % Y_SIZE;
            uint32_t z = idx / Y_SIZE;
            thread_data->morton_codes[code_count++] = morton_encode(x, y, z);
        }
    }
    *thread_data->code_count = code_count;
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

    uint8_t *data = malloc(TOTAL_VOXELS * sizeof(uint8_t));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate data array\n");
        return 1;
    }

    FILE *fp = fopen("c8.raw", "rb");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open c8.raw\n");
        free(data);
        return 1;
    }
    fread(data, sizeof(uint8_t), TOTAL_VOXELS, fp);
    fclose(fp);

    clock_t start_time = clock();

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));
    uint32_t **all_morton_codes = malloc(num_threads * sizeof(uint32_t *));
    size_t *all_code_counts = malloc(num_threads * sizeof(size_t));

    size_t voxels_per_thread = TOTAL_VOXELS / num_threads;
    size_t remainder = TOTAL_VOXELS % num_threads;

    for (int i = 0; i < num_threads; ++i) {
        size_t start_idx = i * voxels_per_thread;
        size_t end_idx = start_idx + voxels_per_thread;
        if (i == num_threads - 1) {
            end_idx += remainder;
        }

        size_t max_codes = (end_idx - start_idx) / 4;
        all_morton_codes[i] = malloc(max_codes * sizeof(uint32_t));
        if (!all_morton_codes[i]) {
            fprintf(stderr, "Error: Failed to allocate Morton codes array\n");
            free(data);
            return 1;
        }

        thread_data[i] = (thread_data_t){
            .thread_id = i,
            .start_idx = start_idx,
            .end_idx = end_idx,
            .data = data,
            .morton_codes = all_morton_codes[i],
            .code_count = &all_code_counts[i]};
        pthread_create(&threads[i], NULL, thread_function, &thread_data[i]);
    }

    size_t total_active_voxels = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        total_active_voxels += all_code_counts[i];
    }

    uint32_t *combined_morton_codes = malloc(total_active_voxels * sizeof(uint32_t));
    size_t offset = 0;
    for (int i = 0; i < num_threads; ++i) {
        memcpy(combined_morton_codes + offset, all_morton_codes[i], all_code_counts[i] * sizeof(uint32_t));
        offset += all_code_counts[i];
        free(all_morton_codes[i]);
    }
    free(all_morton_codes);

    qsort(combined_morton_codes, total_active_voxels, sizeof(uint32_t), compare_uint32);

    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Number of active voxels: %zu\n", total_active_voxels);
    printf("First 10 Morton codes:\n");
    for (size_t i = 0; i < 10 && i < total_active_voxels; ++i) {
        printf("%u\n", combined_morton_codes[i]);
    }
    printf("Processing time with %d threads: %f seconds\n", num_threads, total_time);

    FILE *out_fp = fopen("morton_codes_pthread.txt", "w");
    if (out_fp) {
        for (size_t i = 0; i < total_active_voxels; ++i) {
            fprintf(out_fp, "%u\n", combined_morton_codes[i]);
        }
        fclose(out_fp);
    }

    free(combined_morton_codes);
    free(all_code_counts);
    free(thread_data);
    free(threads);
    free(data);

    return 0;
}
