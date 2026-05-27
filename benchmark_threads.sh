#!/bin/bash
# across different thread countsompv8 vs ompv15performance comparison test
# using real ResNet50 data, 3note

export LD_LIBRARY_PATH="/home/exouser/.appfl/.compressor/SZ3/lib:$LD_LIBRARY_PATH"

DATASET_DIR="/home/exouser/compressor/final/dataset"
THREADS=(1 2 4 8 16)
VERSIONS=("ompv8" "ompv15")
ROUNDS=3

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     ompv8 vs ompv15 performance comparison test (ResNet50notedata)              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "test configuration:"
echo "  • dataset: ResNet50 (266note, ~92MB)"
echo "  • thread count: 1, 2, 4, 8, 16"
echo "  • rounds per configuration: $ROUNDS note"
echo "  • tested version: ompv8 (note), ompv15 (noterowsnote)"
echo ""
echo "════════════════════════════════════════════════════════════════════"

# results file
RESULT_FILE="/tmp/benchmark_threads_results.txt"
> $RESULT_FILE

# build both versions
cd /home/exouser/compressor/final

echo ""
echo "build build ompv8..."
gcc -std=c99 -O3 -fopenmp -march=native \
  -I. -I/home/exouser/.appfl/.compressor/SZ3/include \
  -L/home/exouser/.appfl/.compressor/SZ3/lib \
  test_c_real.c ompv8.c \
  -lSZ3c -lzstd -lz -lm \
  -o test_ompv8 2>&1 | grep -i error

if [ $? -eq 0 ]; then
    echo "FAIL ompv8 buildfailed"
    exit 1
fi
echo "PASS ompv8 buildsucceeded"

echo ""
echo "build build ompv15..."
gcc -std=c99 -O3 -fopenmp -march=native \
  -I. -I/home/exouser/.appfl/.compressor/SZ3/include \
  -L/home/exouser/.appfl/.compressor/SZ3/lib \
  test_c_real.c ompv15.c \
  -lSZ3c -lzstd -lz -lm \
  -o test_ompv15 2>&1 | grep -i error

if [ $? -eq 0 ]; then
    echo "FAIL ompv15 buildfailed"
    exit 1
fi
echo "PASS ompv15 buildsucceeded"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "starting performance test..."
echo "════════════════════════════════════════════════════════════════════"

# test loop
for threads in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$threads
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "config thread count: $threads"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for version in "${VERSIONS[@]}"; do
        echo ""
        echo "  metrics tested version: $version"
        echo "  ────────────────────────────────────────────────────────────"

        times=()

        for round in $(seq 1 $ROUNDS); do
            echo -n "    Round $round: "

            # run test and extract timing
            output=$(./test_${version} 2>&1)

            # extract total time (noteoutputnote "Total time: XXX ms")
            time_ms=$(echo "$output" | grep -oP "Total.*?:\s*\K[0-9]+\.?[0-9]*(?=\s*ms)" | tail -1)

            if [ -z "$time_ms" ]; then
                echo "FAIL could not extract timing"
                time_ms="N/A"
            else
                echo "PASS ${time_ms} ms"
                times+=($time_ms)
            fi
        done

        # compute average
        if [ ${#times[@]} -eq $ROUNDS ]; then
            sum=0
            for t in "${times[@]}"; do
                sum=$(echo "$sum + $t" | bc)
            done
            avg=$(echo "scale=2; $sum / $ROUNDS" | bc)
            echo "    ────────────────────────────────────────────────────────────"
            echo "    trend average time: ${avg} ms"

            # record result
            echo "${threads},${version},${avg}" >> $RESULT_FILE
        fi
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "test complete, generating comparison report..."
echo "════════════════════════════════════════════════════════════════════"

# generate comparison table
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                        note                                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
printf "%-10s | %-15s | %-15s | %-15s | %-15s\n" "Threads" "ompv8 (ms)" "ompv15 (ms)" "Speedup" "Difference"
echo "────────────────────────────────────────────────────────────────────────────────────────"

# read results and compare
for threads in "${THREADS[@]}"; do
    v8_time=$(grep "^${threads},ompv8," $RESULT_FILE | cut -d',' -f3)
    v15_time=$(grep "^${threads},ompv15," $RESULT_FILE | cut -d',' -f3)

    if [ -n "$v8_time" ] && [ -n "$v15_time" ]; then
        speedup=$(echo "scale=4; $v8_time / $v15_time" | bc)
        diff=$(echo "scale=2; (($v15_time - $v8_time) / $v8_time) * 100" | bc)

        # formatted output
        if (( $(echo "$diff > 0" | bc -l) )); then
            diff_str="+${diff}%"
        else
            diff_str="${diff}%"
        fi

        printf "%-10s | %-15s | %-15s | %-15s | %-15s\n" \
            "$threads" "$v8_time" "$v15_time" "${speedup}x" "$diff_str"
    fi
done

echo "────────────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "metrics result interpretation:"
echo "   • Speedup > 1.0: ompv8 note"
echo "   • Speedup < 1.0: ompv15 note"
echo "   • Speedup ~ 1.0: note"
echo ""
echo "saved detailed results saved at: $RESULT_FILE"
echo ""
