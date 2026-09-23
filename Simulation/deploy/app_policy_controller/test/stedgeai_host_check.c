// Host harness for the ST Edge AI backend.
//
// It runs THE SAME policy_net_stedgeai.c that goes into the firmware, driving it exactly
// the way the controller does (policy_reset once, policy_step per frame, the GRU state
// carried inside the state struct), but on the Mac instead of the STM32.
//
// It is compiled and driven by Simulation/deploy/stedgeai_host_check.py. That script
// builds it for x86_64 on purpose: ST ships its host runtime libraries as x86_64-only, so
// the harness must match them (a native arm64 build fails to link, and an x86_64 *driver*
// cannot run on this machine - see the notes in that script).
//
//   usage: stedgeai_host_check <frames.bin> <out.bin> <n_frames>
//     frames.bin : n * (33 + 3) float32 = [raw encoder frame | reference feed-forward]
//     out.bin    : n * 20 float32 - the action (4) then the latent z (16) of each step
//
// The trailing 3 floats are the actor's feed-forward block; they are read in the same row
// and split apart here, exactly as the firmware's build_frame does.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "policy_net_stedgeai.h"     // the production header, not a copy

// Same place as the app (BSS), so nothing here changes stack behaviour relative to the
// vehicle build.
static policy_state_t g_state;

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <frames.bin> <out.bin> <n_frames>\n", argv[0]);
        return 2;
    }
    const long n = strtol(argv[3], NULL, 10);

    FILE *fi = fopen(argv[1], "rb");
    if (!fi) {
        perror("frames.bin");
        return 1;
    }
    FILE *fo = fopen(argv[2], "wb");
    if (!fo) {
        perror("out.bin");
        return 1;
    }

    policy_reset(&g_state);

    float frame[POLICY_ENC_IN_DIM];
    float ref_ff[POLICY_REF_FF_DIM];
    float act[POLICY_ACT_DIM];

    for (long i = 0; i < n; i++) {
        float row[POLICY_ENC_IN_DIM + POLICY_REF_FF_DIM];
        const size_t want = (size_t)POLICY_ENC_IN_DIM + (size_t)POLICY_REF_FF_DIM;
        if (fread(row, sizeof(float), want, fi) != want) {
            fprintf(stderr, "short read at frame %ld\n", i);
            return 1;
        }
        memcpy(frame, row, sizeof(float) * POLICY_ENC_IN_DIM);
        memcpy(ref_ff, row + POLICY_ENC_IN_DIM, sizeof(float) * POLICY_REF_FF_DIM);
        policy_step(&g_state, frame, ref_ff, act);
        if (g_state.errors != 0u) {
            fprintf(stderr, "policy_step failed at frame %ld (rc %d)\n",
                    i, (int)g_state.last_error);
            return 1;
        }
        if (fwrite(act, sizeof(float), POLICY_ACT_DIM, fo) != POLICY_ACT_DIM ||
            fwrite(g_state.z, sizeof(float), POLICY_Z_DIM, fo) != POLICY_Z_DIM) {
            fprintf(stderr, "short write at frame %ld\n", i);
            return 1;
        }
    }

    fclose(fi);
    fclose(fo);
    fprintf(stderr, "ran %ld frames: %u inferences, %u errors\n",
            n, (unsigned)g_state.runs, (unsigned)g_state.errors);
    return 0;
}
