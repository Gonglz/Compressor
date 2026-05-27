#!/bin/bash
# 不同线程数下ompv8 vs ompv15性能对比测试
# 使用ResNet50真实数据，3轮测试

export LD_LIBRARY_PATH="/home/exouser/.appfl/.compressor/SZ3/lib:$LD_LIBRARY_PATH"

DATASET_DIR="/home/exouser/compressor/final/dataset"
THREADS=(1 2 4 8 16)
VERSIONS=("ompv8" "ompv15")
ROUNDS=3

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     ompv8 vs ompv15 性能对比测试 (ResNet50真实数据)              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "测试配置:"
echo "  • 数据集: ResNet50 (266层, ~92MB)"
echo "  • 线程数: 1, 2, 4, 8, 16"
echo "  • 每个配置运行: $ROUNDS 轮"
echo "  • 测试版本: ompv8 (基准), ompv15 (解压缩并行化)"
echo ""
echo "════════════════════════════════════════════════════════════════════"

# 结果文件
RESULT_FILE="/tmp/benchmark_threads_results.txt"
> $RESULT_FILE

# 编译两个版本
cd /home/exouser/compressor/final

echo ""
echo "📦 编译 ompv8..."
gcc -std=c99 -O3 -fopenmp -march=native \
  -I. -I/home/exouser/.appfl/.compressor/SZ3/include \
  -L/home/exouser/.appfl/.compressor/SZ3/lib \
  test_c_real.c ompv8.c \
  -lSZ3c -lzstd -lz -lm \
  -o test_ompv8 2>&1 | grep -i error

if [ $? -eq 0 ]; then
    echo "❌ ompv8 编译失败"
    exit 1
fi
echo "✅ ompv8 编译成功"

echo ""
echo "📦 编译 ompv15..."
gcc -std=c99 -O3 -fopenmp -march=native \
  -I. -I/home/exouser/.appfl/.compressor/SZ3/include \
  -L/home/exouser/.appfl/.compressor/SZ3/lib \
  test_c_real.c ompv15.c \
  -lSZ3c -lzstd -lz -lm \
  -o test_ompv15 2>&1 | grep -i error

if [ $? -eq 0 ]; then
    echo "❌ ompv15 编译失败"
    exit 1
fi
echo "✅ ompv15 编译成功"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "开始性能测试..."
echo "════════════════════════════════════════════════════════════════════"

# 测试循环
for threads in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$threads
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔧 线程数: $threads"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for version in "${VERSIONS[@]}"; do
        echo ""
        echo "  📊 测试版本: $version"
        echo "  ────────────────────────────────────────────────────────────"
        
        times=()
        
        for round in $(seq 1 $ROUNDS); do
            echo -n "    Round $round: "
            
            # 运行测试并提取时间
            output=$(./test_${version} 2>&1)
            
            # 提取总时间 (假设输出格式为 "Total time: XXX ms")
            time_ms=$(echo "$output" | grep -oP "Total.*?:\s*\K[0-9]+\.?[0-9]*(?=\s*ms)" | tail -1)
            
            if [ -z "$time_ms" ]; then
                echo "❌ 无法提取时间"
                time_ms="N/A"
            else
                echo "✅ ${time_ms} ms"
                times+=($time_ms)
            fi
        done
        
        # 计算平均值
        if [ ${#times[@]} -eq $ROUNDS ]; then
            sum=0
            for t in "${times[@]}"; do
                sum=$(echo "$sum + $t" | bc)
            done
            avg=$(echo "scale=2; $sum / $ROUNDS" | bc)
            echo "    ────────────────────────────────────────────────────────────"
            echo "    📈 平均时间: ${avg} ms"
            
            # 记录结果
            echo "${threads},${version},${avg}" >> $RESULT_FILE
        fi
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "测试完成，生成对比报告..."
echo "════════════════════════════════════════════════════════════════════"

# 生成对比表格
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                        性能对比汇总表                                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
printf "%-10s | %-15s | %-15s | %-15s | %-15s\n" "Threads" "ompv8 (ms)" "ompv15 (ms)" "Speedup" "Difference"
echo "────────────────────────────────────────────────────────────────────────────────────────"

# 读取结果并对比
for threads in "${THREADS[@]}"; do
    v8_time=$(grep "^${threads},ompv8," $RESULT_FILE | cut -d',' -f3)
    v15_time=$(grep "^${threads},ompv15," $RESULT_FILE | cut -d',' -f3)
    
    if [ -n "$v8_time" ] && [ -n "$v15_time" ]; then
        speedup=$(echo "scale=4; $v8_time / $v15_time" | bc)
        diff=$(echo "scale=2; (($v15_time - $v8_time) / $v8_time) * 100" | bc)
        
        # 格式化输出
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
echo "📊 结果解读:"
echo "   • Speedup > 1.0: ompv8 更快"
echo "   • Speedup < 1.0: ompv15 更快"
echo "   • Speedup ≈ 1.0: 性能相当"
echo ""
echo "💾 详细结果保存在: $RESULT_FILE"
echo ""
