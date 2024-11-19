#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <limits.h>

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
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t voxels_per_proc = TOTAL_VOXELS / world_size;
    size_t remainder = TOTAL_VOXELS % world_size;

    // Change counts and displs to int arrays
    int *counts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));

    // Assign counts and displacements, ensuring they fit into int
    for (int i = 0; i < world_size; i++) {
        size_t count = voxels_per_proc;
        if (i == world_size - 1) {
            count += remainder;
        }
        if (count > INT_MAX) {
            fprintf(stderr, "Process %d: Count exceeds INT_MAX\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[i] = (int)count;
        displs[i] = (int)(i * voxels_per_proc);
    }

    size_t local_voxel_count = (size_t)counts[world_rank];

    // Allocate local data
    uint8_t *local_data = malloc(local_voxel_count * sizeof(uint8_t));
    if (!local_data) {
        fprintf(stderr, "Process %d: Failed to allocate local_data array\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Root process reads the entire data and scatters it
    if (world_rank == 0) {
        uint8_t *data = malloc(TOTAL_VOXELS * sizeof(uint8_t));
        if (!data) {
            fprintf(stderr, "Process %d: Failed to allocate data array\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read c8.raw
        FILE *fp = fopen("c8.raw", "rb");
        if (!fp) {
            fprintf(stderr, "Process %d: Failed to open c8.raw\n", world_rank);
            free(data);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size_t items_read = fread(data, sizeof(uint8_t), TOTAL_VOXELS, fp);
        fclose(fp);
        if (items_read != TOTAL_VOXELS) {
            fprintf(stderr, "Process %d: Failed to read data from c8.raw\n", world_rank);
            free(data);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Scatter the data
        MPI_Scatterv(data, counts, displs, MPI_UNSIGNED_CHAR, local_data, counts[world_rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        free(data);
    } else {
        // Other processes receive their portion
        MPI_Scatterv(NULL, NULL, NULL, MPI_UNSIGNED_CHAR, local_data, counts[world_rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    // Start timing
    double start_time = MPI_Wtime();

    // Process local data
    size_t max_codes = 1000000; // Initial size, adjust as needed
    uint32_t *morton_codes = malloc(max_codes * sizeof(uint32_t));
    if (!morton_codes) {
        fprintf(stderr, "Process %d: Failed to allocate morton_codes\n", world_rank);
        free(local_data);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t code_count = 0;

    size_t start_idx = (size_t)displs[world_rank];
    for (size_t i = 0; i < local_voxel_count; ++i) {
        uint8_t value = local_data[i];
        if (value > THRESHOLD) {
            // Compute global index
            size_t idx = start_idx + i;
            // Compute x, y, z from idx
            size_t temp_idx = idx;
            uint32_t x = temp_idx % X_SIZE;
            temp_idx /= X_SIZE;
            uint32_t y = temp_idx % Y_SIZE;
            uint32_t z = temp_idx / Y_SIZE;

            uint32_t morton_code = morton_encode(x, y, z);

            if (code_count >= max_codes) {
                // Need to reallocate morton_codes
                size_t new_size = max_codes * 2;
                uint32_t *new_array = realloc(morton_codes, new_size * sizeof(uint32_t));
                if (!new_array) {
                    fprintf(stderr, "Process %d: Failed to reallocate morton_codes\n", world_rank);
                    free(morton_codes);
                    free(local_data);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                morton_codes = new_array;
                max_codes = new_size;
            }
            morton_codes[code_count++] = morton_code;
        }
    }

    // Convert code_count to int for MPI
    if (code_count > INT_MAX) {
        fprintf(stderr, "Process %d: code_count exceeds INT_MAX\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int code_count_int = (int)code_count;

    // Gather code counts
    int *recv_counts = NULL;
    int *recv_displs = NULL;
    if (world_rank == 0) {
        recv_counts = malloc(world_size * sizeof(int));
        recv_displs = malloc(world_size * sizeof(int));
    }
    MPI_Gather(&code_count_int, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather morton_codes
    uint32_t *all_morton_codes = NULL;
    if (world_rank == 0) {
        int total_codes = 0;
        recv_displs[0] = 0;
        for (int i = 0; i < world_size; ++i) {
            total_codes += recv_counts[i];
            if (i > 0) {
                recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            }
        }
        all_morton_codes = malloc(total_codes * sizeof(uint32_t));
    }

    MPI_Gatherv(morton_codes, code_count_int, MPI_UINT32_T, all_morton_codes, recv_counts, recv_displs, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Root process sorts the combined array
    if (world_rank == 0) {
        int total_codes = 0;
        for (int i = 0; i < world_size; ++i) {
            total_codes += recv_counts[i];
        }

        // Sort all_morton_codes[]
        qsort(all_morton_codes, total_codes, sizeof(uint32_t), compare_uint32);

        // End timing
        double end_time = MPI_Wtime();
        printf("Number of active voxels: %d\n", total_codes);

        // Output first 10 Morton codes
        printf("First 10 Morton codes:\n");
        for (int i = 0; i < 10 && i < total_codes; ++i) {
            printf("%u\n", all_morton_codes[i]);
        }

        // Verify if all_morton_codes[] is sorted
        int is_sorted = 1;
        for (int i = 1; i < total_codes; ++i) {
            if (all_morton_codes[i - 1] > all_morton_codes[i]) {
                is_sorted = 0;
                fprintf(stderr, "Array is not sorted at index %d\n", i);
                break;
            }
        }
        if (is_sorted) {
            printf("Morton codes are correctly sorted.\n");
        } else {
            printf("Morton codes are NOT correctly sorted.\n");
        }

        // Save Morton codes to file
        FILE *out_fp = fopen("morton_codes_mpi.txt", "w");
        if (out_fp) {
            for (int i = 0; i < total_codes; ++i) {
                fprintf(out_fp, "%u\n", all_morton_codes[i]);
            }
            fclose(out_fp);
            printf("Morton codes saved to morton_codes_mpi.txt\n");
        } else {
            fprintf(stderr, "Error: Failed to open output file for writing.\n");
        }

        printf("Processing time with %d processes: %f seconds\n", world_size, end_time - start_time);

        free(all_morton_codes);
        free(recv_counts);
        free(recv_displs);
    }

    free(morton_codes);
    free(local_data);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
