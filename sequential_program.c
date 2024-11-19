#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define X_SIZE 1024
#define Y_SIZE 1024
#define Z_SIZE 314

#define TOTAL_VOXELS ((size_t)X_SIZE * Y_SIZE * Z_SIZE)
#define THRESHOLD 25
#define BITS_PER_COORD 10

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

int main(int argc, char *argv[]) {
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

    // Initialize morton_codes[]
    size_t max_codes = 1000000; // Initial size, will expand as needed
    uint32_t *morton_codes = malloc(max_codes * sizeof(uint32_t));
    if (!morton_codes) {
        fprintf(stderr, "Error: Failed to allocate morton_codes array\n");
        free(data);
        return 1;
    }
    size_t code_count = 0;

    // Variables to track coordinate ranges
    uint32_t min_x = X_SIZE, min_y = Y_SIZE, min_z = Z_SIZE;
    uint32_t max_x = 0, max_y = 0, max_z = 0;

    // Process data
    for (size_t i = 0; i < TOTAL_VOXELS; ++i) {
        uint8_t value = data[i];
        if (value > THRESHOLD) {
            // Compute x, y, z from index i
            size_t idx = i;
            uint32_t x = idx % X_SIZE;
            idx /= X_SIZE;
            uint32_t y = idx % Y_SIZE;
            uint32_t z = idx / Y_SIZE;

            // Update min and max coordinates
            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
            if (z < min_z) min_z = z;
            if (z > max_z) max_z = z;

            uint32_t morton_code = morton_encode(x, y, z);

            // Check if morton_codes[] needs to be reallocated
            if (code_count >= max_codes) {
                size_t new_size = max_codes * 2;
                uint32_t *new_array = realloc(morton_codes, new_size * sizeof(uint32_t));
                if (!new_array) {
                    fprintf(stderr, "Error: Failed to reallocate morton_codes array\n");
                    free(morton_codes);
                    free(data);
                    return 1;
                }
                morton_codes = new_array;
                max_codes = new_size;
            }
            morton_codes[code_count++] = morton_code;
        }
    }

    // Sort morton_codes[]
    qsort(morton_codes, code_count, sizeof(uint32_t), compare_uint32);

    // Timing ends here
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Output number of active voxels
    printf("Number of active voxels: %zu\n", code_count);

    // Output coordinate ranges
    printf("Coordinate ranges:\n");
    printf("X: min = %u, max = %u\n", min_x, max_x);
    printf("Y: min = %u, max = %u\n", min_y, max_y);
    printf("Z: min = %u, max = %u\n", min_z, max_z);

    // Output first 10 Morton codes
    printf("First 10 Morton codes:\n");
    for (size_t i = 0; i < 10 && i < code_count; ++i) {
        uint32_t morton_code = morton_codes[i];
        printf("%u\n", morton_code);
    }

    // Verify if morton_codes[] is sorted
    int is_sorted = 1;
    for (size_t i = 1; i < code_count; ++i) {
        if (morton_codes[i - 1] > morton_codes[i]) {
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
    FILE *out_fp = fopen("morton_codes_seq.txt", "w");
    if (out_fp) {
        for (size_t i = 0; i < code_count; ++i) {
            fprintf(out_fp, "%u\n", morton_codes[i]);
        }
        fclose(out_fp);
        printf("Morton codes saved to morton_codes_seq.txt\n");
    } else {
        fprintf(stderr, "Error: Failed to open output file for writing.\n");
    }

    // Output processing time
    printf("Processing time (sequential): %f seconds\n", total_time);

    // Free resources
    free(morton_codes);
    free(data);

    return 0;
}
