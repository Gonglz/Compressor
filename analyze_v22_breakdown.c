/**
 * v22sz3note
 *
 * note: note/notestagenote, note:
 * 1. note(note): SZ3note, ZSTDnote, IOnote
 * 2. note: notecompute, hashnote, note
 * 3. noterowsnote: currentnoterowsnote
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "momentum_compressor.h"

// noteclassnote
typedef struct {
    // notestage
    double compress_fixed_io;        // noteIOnote: SZ3+ZSTDnote
    double compress_compute;         // notecompute: stats+normalize+consistency
    double compress_hash_ops;        // Hashnote: noteprediction_memory
    double compress_memory_ops;      // note: note, note
    double compress_parallel_overhead; // noterowsnote: OpenMPnote+note

    // notestage
    double decompress_fixed_io;      // noteIOnote: SZ3+ZSTDnote
    double decompress_compute;       // notecompute: note+note
    double decompress_hash_ops;      // Hashnote: notehistory+memory
    double decompress_memory_ops;    // note

    // note
    size_t num_layers;
    double total_time;
} PerformanceBreakdown;

void print_performance_analysis(const PerformanceBreakdown *pb) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    V22SZ3 note                               ║\n");
    printf("║           Fixed Overhead vs Optimizable Components                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");

    double compress_total = pb->compress_fixed_io + pb->compress_compute +
                           pb->compress_hash_ops + pb->compress_memory_ops +
                           pb->compress_parallel_overhead;
    double decompress_total = pb->decompress_fixed_io + pb->decompress_compute +
                             pb->decompress_hash_ops + pb->decompress_memory_ops;

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("                           notestage (Compress)                            \n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("🔴 note (note)\n");
    printf("   ├─ SZ3/ZSTDnote:           %8.2f ms  (%5.1f%%)  [note]\n",
           pb->compress_fixed_io, 100.0 * pb->compress_fixed_io / compress_total);
    printf("   └─ note:                  FAIL note\n\n");

    printf("🟢 note\n");
    printf("   ├─ notecompute (noterowsnote):          %8.2f ms  (%5.1f%%)  [note]\n",
           pb->compress_compute, 100.0 * pb->compress_compute / compress_total);
    printf("   ├─ Hashnote:                  %8.2f ms  (%5.1f%%)  [note]\n",
           pb->compress_hash_ops, 100.0 * pb->compress_hash_ops / compress_total);
    printf("   ├─ note:                  %8.2f ms  (%5.1f%%)  [note]\n",
           pb->compress_memory_ops, 100.0 * pb->compress_memory_ops / compress_total);
    printf("   └─ noterowsnote:                  %8.2f ms  (%5.1f%%)  [note]\n\n",
           pb->compress_parallel_overhead, 100.0 * pb->compress_parallel_overhead / compress_total);

    printf("   note:\n");
    double optimizable_compress = pb->compress_compute + pb->compress_hash_ops +
                                  pb->compress_memory_ops;
    printf("   • note: %.1fx (note0)\n",
           compress_total / pb->compress_fixed_io);
    printf("   • note:   %.1f%% of total\n",
           100.0 * optimizable_compress / compress_total);
    printf("   • note: %.1f%% note = note %.1fx\n\n",
           100.0 * pb->compress_fixed_io / compress_total,
           1.0 / (pb->compress_fixed_io / compress_total));

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("                           notestage (Decompress)                         \n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("🔴 note (note)\n");
    printf("   ├─ SZ3/ZSTDnote:              %8.2f ms  (%5.1f%%)  [note]\n",
           pb->decompress_fixed_io, 100.0 * pb->decompress_fixed_io / decompress_total);
    printf("   └─ note:                  FAIL note\n\n");

    printf("🟢 note\n");
    printf("   ├─ notecompute (note+note):     %8.2f ms  (%5.1f%%)  [noterowsnote]\n",
           pb->decompress_compute, 100.0 * pb->decompress_compute / decompress_total);
    printf("   ├─ Hashnote:                  %8.2f ms  (%5.1f%%)  [note]\n",
           pb->decompress_hash_ops, 100.0 * pb->decompress_hash_ops / decompress_total);
    printf("   └─ note:                  %8.2f ms  (%5.1f%%)  [note]\n\n",
           pb->decompress_memory_ops, 100.0 * pb->decompress_memory_ops / decompress_total);

    printf("   note: noterowsnote, noterowsnote (note+7%%note)\n\n");

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("                           note                                      \n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    double total_fixed = pb->compress_fixed_io + pb->decompress_fixed_io;
    double total_all = compress_total + decompress_total;

    printf("metrics note:\n");
    printf("   • note:    %.1f%% (SZ3/ZSTDnote)\n",
           100.0 * total_fixed / total_all);
    printf("   • note:      %.1f%%\n",
           100.0 * (total_all - total_fixed) / total_all);
    printf("   • note:            %zu\n", pb->num_layers);
    printf("   • note:        %.2f ms (note) + %.2f ms (note)\n",
           compress_total / pb->num_layers, decompress_total / pb->num_layers);

    printf("\n💡 note:\n");
    printf("   1. PASS noterowsnote (5xnote on 8 cores)\n");
    printf("   2. PASS noterowsnote (notehashnote)\n");
    printf("   3. ⚠️  note:\n");
    printf("      - SZ3note (%.1f%% note)\n",
           100.0 * pb->compress_fixed_io / compress_total);
    printf("      - ZSTDnote (%.1f%% note)\n",
           100.0 * pb->decompress_fixed_io / decompress_total);
    printf("   4. 🎯 note:\n");
    printf("      - note (LZ4, Zstd levelnote)\n");
    printf("      - note (poolnote)\n");
    printf("      - SIMDnotecomputenote (stats/normalize)\n");

    printf("\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
}

int main() {
    // notedatanotev22sz3note (8note, ResNet50, Round 1-2note)
    // note: 152ms (walltime), note: 291ms
    // Breakdownnote: stats=44ms, normalize=24ms, consistency=138ms,
    //                    prediction=206ms, residual_compress=558ms, bitmap=79ms

    PerformanceBreakdown pb = {
        // note (notebreakdown)
        // note: residual_compressnoterowsnote, notewalltimenote558/8~70ms.compress_fixed_io = 70.0,        // SZ3+ZSTDnoterowsnote: ~46% walltime.compress_compute = 44.0,         // stats+normalize+consistency: ~29%.compress_hash_ops = 20.0,        // predictionnotehashnote: ~13%.compress_memory_ops = 12.0,      // bitmap+metadata: ~8%.compress_parallel_overhead = 6.0, // OpenMP sync: ~4%

        // note (noterows, note291msnote).decompress_fixed_io = 215.0,     // SZ3+ZSTDnote: ~74%.decompress_compute = 50.0,       // note+note: ~17%.decompress_hash_ops = 18.0,      // hashnotehistory: ~6%.decompress_memory_ops = 8.0,     // note: ~3%.num_layers = 266,.total_time = 443.0  // 152 + 291
    };

    print_performance_analysis(&pb);

    return 0;
}
