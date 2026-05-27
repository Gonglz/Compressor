#!/bin/bash
# ompv8 vs ompv15 性能对比测试 - 使用真实ResNet50数据

export LD_LIBRARY_PATH="/home/exouser/.appfl/.compressor/SZ3/lib:$LD_LIBRARY_PATH"

DATASET_DIR="/home/exouser/compressor/final/dataset/resnet50"
THREADS=(1 2 4 8 16)
ROUNDS="round_0_client_0.bin round_1_client_0.bin round_2_client_0.bin"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     ompv8 vs ompv15 性能对比测试 (ResNet50真实数据)              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "测试配置:"
echo "  • 数据集: ResNet50 (266层, ~92MB)"
echo "  • 线程数: 1, 2, 4, 8, 16"
echo "  • 每个配置运行: 3轮"
echo "  • 测试版本: ompv8, ompv15"
echo ""

# 检查数据文件
cd $DATASET_DIR
missing=0
for round in $ROUNDS; do
    if [ ! -f "$round" ]; then
        echo "⚠️  数据文件不存在: $round"
        missing=1
    fi
done

if [ $missing -eq 1 ]; then
    echo "❌ 请确保所有ResNet50数据文件存在"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "开始性能测试..."
echo "════════════════════════════════════════════════════════════════════"

# 结果数组
declare -A results_v8
declare -A results_v15

cd /home/exouser/compressor/final

for threads in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$threads
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔧 线程数: $threads"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 测试 ompv8
    echo "  📊 ompv8:"
    output=$(./benchmark_ompv8 $DATASET_DIR/round_1_resnet50.bin \
                               $DATASET_DIR/round_2_resnet50.bin \
                               $DATASET_DIR/round_3_resnet50.bin 2>&1)
    v8_time=$(echo "$output" | grep "BENCHMARK_RESULT:" | awk '{print $2}')
    
    if [ -n "$v8_time" ]; then
        echo "      ✅ ${v8_time}"
        results_v8[$threads]=$v8_time
    else
        echo "      ❌ 测试失败"
        results_v8[$threads]="N/A"
    fi
    
    # 测试 ompv15
    echo "  📊 ompv15:"
    output=$(./benchmark_ompv15 $DATASET_DIR/round_1_resnet50.bin \
                                $DATASET_DIR/round_2_resnet50.bin \
                                $DATASET_DIR/round_3_resnet50.bin 2>&1)
    v15_time=$(echo "$output" | grep "BENCHMARK_RESULT:" | awk '{print $2}')
    
    if [ -n "$v15_time" ]; then
        echo "      ✅ ${v15_time}"
        results_v15[$threads]=$v15_time
    else
        echo "      ❌ 测试失败"
        results_v15[$threads]="N/A"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "生成对比报告..."
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                        性能对比汇总表                                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
printf "%-10s | %-15s | %-15s | %-15s | %-15s\n" "Threads" "ompv8 (ms)" "ompv15 (ms)" "Speedup" "Difference"
echo "────────────────────────────────────────────────────────────────────────────────────────"

for threads in "${THREADS[@]}"; do
    v8_time=${results_v8[$threads]}
    v15_time=${results_v15[$threads]}
    
    if [ "$v8_time" != "N/A" ] && [ "$v15_time" != "N/A" ]; then
        # 移除 "ms" 后缀
        v8_num=$(echo $v8_time | sed 's/ms//')
        v15_num=$(echo $v15_time | sed 's/ms//')
        
        speedup=$(echo "scale=4; $v8_num / $v15_num" | bc)
        diff=$(echo "scale=2; (($v15_num - $v8_num) / $v8_num) * 100" | bc)
        
        # 格式化差异
        if (( $(echo "$diff > 0" | bc -l) )); then
            diff_str="+${diff}%"
            status="⚠️"
        elif (( $(echo "$diff < -1" | bc -l) )); then
            diff_str="${diff}%"
            status="✅"
        else
            diff_str="${diff}%"
            status="≈"
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
echo "📊 符号说明:"
echo "   ✅ v15更快 (差异 < -1%)"
echo "   ≈  性能相当 (差异 -1% ~ 0%)"
echo "   ⚠️  v8更快 (差异 > 0%)"
echo ""
echo "💡 结论:"
echo "   • Speedup > 1.0: ompv8 比 ompv15 快"
echo "   • Speedup < 1.0: ompv15 比 ompv8 快"  
echo "   • Speedup ≈ 1.0: 两者性能相当"
echo ""
