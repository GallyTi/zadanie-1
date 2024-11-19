#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 10

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s file1 file2 [file3 ...]\n", argv[0]);
        return 1;
    }

    int num_files = argc - 1;
    if (num_files > MAX_FILES) {
        fprintf(stderr, "Error: Maximum number of files to compare is %d\n", MAX_FILES);
        return 1;
    }

    FILE *fps[MAX_FILES];
    for (int i = 0; i < num_files; i++) {
        fps[i] = fopen(argv[i + 1], "r");
        if (!fps[i]) {
            fprintf(stderr, "Error: Failed to open file %s\n", argv[i + 1]);
            // Close previously opened files
            for (int j = 0; j < i; j++) {
                fclose(fps[j]);
            }
            return 1;
        }
    }

    unsigned int codes[MAX_FILES];
    size_t line = 1;
    int identical = 1;
    int eof_flags[MAX_FILES];
    memset(eof_flags, 0, sizeof(eof_flags));

    while (1) {
        int num_eof = 0;
        int read_success = 1;

        // Attempt to read from each file
        for (int i = 0; i < num_files; i++) {
            if (eof_flags[i]) {
                num_eof++;
                continue;
            }
            if (fscanf(fps[i], "%u", &codes[i]) != 1) {
                eof_flags[i] = 1;
                num_eof++;
            }
        }

        if (num_eof == num_files) {
            // All files have reached EOF
            break;
        } else if (num_eof > 0) {
            // Some files have reached EOF before others
            printf("Files have different lengths.\n");
            identical = 0;
            break;
        }

        // Compare codes
        for (int i = 1; i < num_files; i++) {
            if (codes[i] != codes[0]) {
                printf("Difference at line %zu:\n", line);
                for (int j = 0; j < num_files; j++) {
                    printf("  %s: %u\n", argv[j + 1], codes[j]);
                }
                identical = 0;
                goto cleanup;
            }
        }
        line++;
    }

cleanup:
    for (int i = 0; i < num_files; i++) {
        fclose(fps[i]);
    }

    if (identical) {
        printf("Files are identical.\n");
    } else {
        printf("Files are NOT identical.\n");
    }

    return 0;
}


/*
Kompilácia
gcc -o sequential_program sequential_program.c
gcc -pthread -o pthread_program pthread_program.c
mpicc -o mpi_program mpi_program.c
gcc -o compare_results compare_results.c

# Spustenie sekvenčného programu
./sequential_program

# Spustenie pthread programu s 4 vláknami
./pthread_program 4

# Spustenie MPI programu s 4 procesmi
mpirun -np 4 ./mpi_program

# Porovnanie sekvenčného a pthread výsledku
./compare_results morton_codes_seq.txt morton_codes_pthread.txt

# Porovnanie všetkých troch výsledkov
./compare_results morton_codes_seq.txt morton_codes_pthread.txt morton_codes_mpi.txt
*/