// Host harness for the ONBOARD REFERENCE SAMPLER (reference.c + the baked tables).
//
// WHY THIS EXISTS
// ---------------
// `policy_host_check.py` covers the policy: frame in, action out. This covers the OTHER
// half of the deployed observation - the part that turns a baked table into the reference
// ERRORS *and* the feed-forward block the actor now consumes. Nothing else compares
// reference.c against the simulator, so without this a sign error in ref_feed_forward
// (g subtracted instead of added), a wrong row offset, or a broken relocation would reach
// the vehicle and only show up as odd behaviour in the air.
//
// It is driven by `gen_references.py`, which rebuilds the same relocation with
// `live_target.ShiftedTrajectory` - the reference the SIMULATOR hands the policy - so a
// pass here means the on-board reference describes the same commanded motion the policy
// was trained against, tick for tick.
//
//   usage: ref_check out.bin
//     out.bin : for every kind in REF_KIND_* order, REF_LEN[kind] rows of 21 float32:
//                 [ p(3) | v(3) | R(9, row-major) | w(3) | ff(3) ]
//               followed by REF_CHECK_HOLD_ROWS rows of the same shape for HOLD mode.
//
// The launch pose / heading / tick below are mirrored by the driver. They match the
// constants `gen_references.check_table` already uses for its own relocation check, so
// both halves of the generator agree about what "somewhere else entirely" means.

#include <stdio.h>
#include <string.h>

#include "reference.h"
#include "generated/policy_weights.h"     // POLICY_SIM_GRAVITY: the exported sim constant

#define REF_CHECK_P0 { 0.41f, -0.33f, 1.37f }
#define REF_CHECK_YAW0 0.7f
#define REF_CHECK_START_STEP 1234u        // 100 Hz ticks -> t0 = 12.34 s
#define REF_CHECK_HOLD_ROWS 8

static int write_row(FILE *fo, const float p[3], const float v[3], const float R[9],
                     const float w[3], const float ff[3])
{
    float row[21];
    int o = 0;
    for (int i = 0; i < 3; i++) {
        row[o++] = p[i];
    }
    for (int i = 0; i < 3; i++) {
        row[o++] = v[i];
    }
    for (int i = 0; i < 9; i++) {
        row[o++] = R[i];
    }
    for (int i = 0; i < 3; i++) {
        row[o++] = w[i];
    }
    for (int i = 0; i < 3; i++) {
        row[o++] = ff[i];
    }
    return (fwrite(row, sizeof(float), 21u, fo) == 21u) ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s out.bin\n", argv[0]);
        return 2;
    }
    FILE *fo = fopen(argv[1], "wb");
    if (fo == NULL) {
        perror("out.bin");
        return 1;
    }

    const float p0[3] = REF_CHECK_P0;
    ref_state_t ref;
    uint32_t rows = 0;

    // -- every baked manoeuvre, launched from the fixed test pose ------------------------
    for (uint8_t kind = 0; kind < ref_kind_count(); kind++) {
        ref_play_launch(&ref, kind, p0, REF_CHECK_YAW0, REF_CHECK_START_STEP);
        const uint32_t len = ref_kind_len(kind);
        for (uint32_t k = 0; k < len; k++) {
            float p[3], v[3], R[9], w[3], a[3], ff[3];
            ref_sample(&ref, REF_CHECK_START_STEP + k, p, v, R, w, a);
            ref_feed_forward(a, POLICY_SIM_GRAVITY, ff);
            if (write_row(fo, p, v, R, w, ff) != 0) {
                fprintf(stderr, "short write at %s row %u\n", ref_kind_name(kind), k);
                fclose(fo);
                return 1;
            }
            rows++;
        }
    }

    // -- HOLD: a stationary level hover whose feed-forward is exactly [0, 0, g] ---------
    ref_hold_sync(&ref, p0, REF_CHECK_YAW0);
    for (uint32_t k = 0; k < REF_CHECK_HOLD_ROWS; k++) {
        float p[3], v[3], R[9], w[3], a[3], ff[3];
        ref_sample(&ref, 100u + k, p, v, R, w, a);
        ref_feed_forward(a, POLICY_SIM_GRAVITY, ff);
        if (write_row(fo, p, v, R, w, ff) != 0) {
            fprintf(stderr, "short write in HOLD\n");
            fclose(fo);
            return 1;
        }
        rows++;
    }

    fclose(fo);
    fprintf(stderr, "ref_check: %u rows, %u kinds, gravity %.2f m/s^2\n",
            rows, (unsigned)ref_kind_count(), (double)POLICY_SIM_GRAVITY);
    return 0;
}
