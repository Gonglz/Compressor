#!/bin/bash
# ompv8 vs ompv15 performance comparison test - noteResNet50data

export LD_LIBRARY_PATH="/home/exouser/.appfl/.compressor/SZ3/lib:$LD_LIBRARY_PATH"

DATASET_DIR="/home/exouser/compressor/final/dataset/resnet50"
THREADS=(1 2 4 8 16)
ROUNDS="round_0_client_0.bin round_1_client_0.bin round_2_client_0.bin"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     ompv8 vs ompv15 performance comparison test (ResNet50notedata)              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "test configuration:"
echo "  • dataset: ResNet50 (266note, ~92MB)"
echo "  • thread count: 1, 2, 4, 8, 16"
echo "  • rounds per configuration: 3note"
echo "  • tested version: ompv8, ompv15"
echo ""

# notedatafile
cd $DATASET_DIR
missing=0
for round in $ROUNDS; do
    if [! -f "$round" ]; then
        echo "⚠️  datafilenote: $round"
        missing=1
    fi
done

if [ $missing -eq 1 ]; then
    echo "FAIL noteResNet50datafilenote"
    exit 1
fi

echo "PASS datafilenote"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "starting performance test..."
echo "════════════════════════════════════════════════════════════════════"

# resultnote
declare -A results_v8
declare -A results_v15

cd /home/exouser/compressor/final

for threads in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$threads
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "config thread count: $threads"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # note ompv8
    echo "  metrics ompv8:"
    output=$(./benchmark_ompv8 $DATASET_DIR/round_1_resnet50.bin \
                               $DATASET_DIR/round_2_resnet50.bin \
                               $DATASET_DIR/round_3_resnet50.bin 2>&1)
    v8_time=$(echo "$output" | grep "BENCHMARK_RESULT:" | awk '{print $2}')

    if [ -n "$v8_time" ]; then
        echo "      PASS ${v8_time}"
        results_v8[$threads]=$v8_time
    else
        echo "      FAIL notefailed"
        results_v8[$threads]="N/A"
    fi

    # note ompv15
    echo "  metrics ompv15:"
    output=$(./benchmark_ompv15 $DATASET_DIR/round_1_resnet50.bin \
                                $DATASET_DIR/round_2_resnet50.bin \
                                $DATASET_DIR/round_3_resnet50.bin 2>&1)
    v15_time=$(echo "$output" | grep "BENCHMARK_RESULT:" | awk '{print $2}')

    if [ -n "$v15_time" ]; then
        echo "      PASS ${v15_time}"
        results_v15[$threads]=$v15_time
    else
        echo "      FAIL notefailed"
        results_v15[$threads]="N/A"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "generating comparison report..."
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                        note                                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
printf "%-10s | %-15s | %-15s | %-15s | %-15s\n" "Threads" "ompv8 (ms)" "ompv15 (ms)" "Speedup" "Difference"
echo "────────────────────────────────────────────────────────────────────────────────────────"

for threads in "${THREADS[@]}"; do
    v8_time=${results_v8[$threads]}
    v15_time=${results_v15[$threads]}

    if [ "$v8_time"!= "N/A" ] && [ "$v15_time"!= "N/A" ]; then
        # note "ms" note
        v8_num=$(echo $v8_time | sed 's/ms//')
        v15_num=$(echo $v15_time | sed 's/ms//')

        speedup=$(echo "scale=4; $v8_num / $v15_num" | bc)
        diff=$(echo "scale=2; (($v15_num - $v8_num) / $v8_num) * 100" | bc)

        # note
        if (( $(echo "$diff > 0" | bc -l) )); then
            diff_str="+${diff}%"
            status="⚠️"
        elif (( $(echo "$diff < -1" | bc -l) )); then
            diff_str="${diff}%"
            status="PASS"
        else
            diff_str="${diff}%"
            status="~"
        fi

        printf "%-10s | %-15s | %-15s | %-15s | %-15s %s\n" \
            "$threads" "$v8_time" "$v15_time" "${speedup}x" "$diff_str" "$status"
    else
        printf "%-10s | %-15s | %-15s | %-15s | %-15s\n" \
            "$threads" "$v8_time" "$v15_time" "N/A" "N/A"
    fi
done

echo "────────────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "metrics notedescription:"
echo "   PASS v15note (note < -1%)"
echo "   ~  note (note -1% ~ 0%)"
echo "   ⚠️  v8note (note > 0%)"
echo ""
echo "💡 note:"
echo "   • Speedup > 1.0: ompv8 note ompv15 note"
echo "   • Speedup < 1.0: ompv15 note ompv8 note"
echo "   • Speedup ~ 1.0: note"
echo ""
