#pragma once

#define benchmark_to_file false

#define INITTIMER            \
    float milliseconds = 0;  \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);  \
    FILE *fp;
#define START cudaEventRecord(start);
#define END_wo_print            \
    cudaEventRecord(stop);      \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&milliseconds, start, stop);
#define END(x)                                                        \
    cudaEventRecord(stop);                                            \
    cudaEventSynchronize(stop);                                       \
    cudaEventElapsedTime(&milliseconds, start, stop);                 \
    printf("[+] %s Elapsed Time: %f ms\n", x, milliseconds);          \
    if (benchmark_to_file)                                            \
    {                                                                 \
        fp = fopen("benchmark", "a+");                                \
        fprintf(fp, "[+] %s Elapsed Time: %f ms\n", x, milliseconds); \
        fclose(fp);                                                   \
    }
