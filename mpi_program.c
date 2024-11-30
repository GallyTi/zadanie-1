#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <string.h>
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

    int *counts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));

    // Assign counts and displacements
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

    // Read data in parallel using MPI I/O
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "c8.raw", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset offset = (MPI_Offset)displs[world_rank];
    MPI_Status status;
    MPI_File_read_at(fh, offset, local_data, counts[world_rank], MPI_UNSIGNED_CHAR, &status);
    MPI_File_close(&fh);

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // First pass: Count active voxels
    size_t active_voxels = 0;
    for (size_t i = 0; i < local_voxel_count; ++i) {
        if (local_data[i] > THRESHOLD) {
            ++active_voxels;
        }
    }

    // Allocate morton_codes based on active_voxels
    uint32_t *morton_codes = malloc(active_voxels * sizeof(uint32_t));
    if (!morton_codes) {
        fprintf(stderr, "Process %d: Failed to allocate morton_codes\n", world_rank);
        free(local_data);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Second pass: Compute Morton codes
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
            morton_codes[code_count++] = morton_code;
        }
    }

    // Sort morton_codes locally
    qsort(morton_codes, code_count, sizeof(uint32_t), compare_uint32);

    // Gather code counts
    int *recv_counts = NULL;
    int *recv_displs = NULL;
    if (world_rank == 0) {
        recv_counts = malloc(world_size * sizeof(int));
        recv_displs = malloc(world_size * sizeof(int));
    }

    int code_count_int = (int)code_count;
    MPI_Gather(&code_count_int, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    // Gather sorted morton_codes to root
    MPI_Gatherv(morton_codes, code_count_int, MPI_UINT32_T, all_morton_codes, recv_counts, recv_displs, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Root process merges the sorted arrays
    if (world_rank == 0) {
        int total_codes = 0;
        for (int i = 0; i < world_size; ++i) {
            total_codes += recv_counts[i];
        }

        // Initialize positions and ends for k-way merge
        int *positions = malloc(world_size * sizeof(int));
        int *ends = malloc(world_size * sizeof(int));
        for (int i = 0; i < world_size; ++i) {
            positions[i] = recv_displs[i];
            ends[i] = recv_displs[i] + recv_counts[i];
        }

        uint32_t *merged_codes = malloc(total_codes * sizeof(uint32_t));

        // K-way merge
        for (int k = 0; k < total_codes; ++k) {
            uint32_t min_value = UINT32_MAX;
            int min_index = -1;
            for (int i = 0; i < world_size; ++i) {
                if (positions[i] < ends[i]) {
                    uint32_t value = all_morton_codes[positions[i]];
                    if (value < min_value) {
                        min_value = value;
                        min_index = i;
                    }
                }
            }
            if (min_index == -1) {
                fprintf(stderr, "Error during merging: no minimum found\n");
                break;
            }
            merged_codes[k] = min_value;
            positions[min_index]++;
        }

        // End timing
        double end_time = MPI_Wtime();

        printf("Number of active voxels: %d\n", total_codes);

        // Output first 10 Morton codes
        printf("First 10 Morton codes:\n");
        for (int i = 0; i < 10 && i < total_codes; ++i) {
            printf("%u\n", merged_codes[i]);
        }

        // Verify if merged_codes[] is sorted
        int is_sorted = 1;
        for (int i = 1; i < total_codes; ++i) {
            if (merged_codes[i - 1] > merged_codes[i]) {
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
                fprintf(out_fp, "%u\n", merged_codes[i]);
            }
            fclose(out_fp);
            printf("Morton codes saved to morton_codes_mpi.txt\n");
        } else {
            fprintf(stderr, "Error: Failed to open output file for writing.\n");
        }

        printf("Processing time with %d processes: %f seconds\n", world_size, end_time - start_time);

        free(merged_codes);
        free(positions);
        free(ends);
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
