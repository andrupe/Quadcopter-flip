// Host harness: run the DEPLOYED C forward pass over a stream of encoder frames.
//
// The point of this program is that it links the same policy_net.c that the Crazyflie
// firmware will link, so the check performed by Simulation/deploy/policy_host_check.py
// (C vs torch) is evidence about the flight code, not about a reimplementation of it.
//
//   usage: host_check frames.bin out.bin
//     frames.bin : N x (33 + 3) float32 = [raw encoder frame | reference feed-forward]
//     out.bin    : N x 20 float32  (4 action + 16 z per frame)
//
// The frame half is the encoder's input contract ([o_t(29) | aux(4)]); the 3 trailing floats
// are the actor's feed-forward block (a_ref + g*e_z, world). They are read together and
// passed SEPARATELY to policy_step, which is what the firmware does.
//
// The stream is fed through ONE policy_state_t from the reset state, i.e. exactly how the
// firmware runs it: the GRU carries its hidden state across the whole flight.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "policy_net.h"

// One input row -> (frame, ref_ff). Returns 0 on success, non-zero at end-of-file or on a
// truncated row (a partial row must not be mistaken for a valid step).
static int read_step(FILE *fin, float frame[POLICY_ENC_IN_DIM],
                     float ref_ff[POLICY_REF_FF_DIM])
{
    float row[POLICY_ENC_IN_DIM + POLICY_REF_FF_DIM];
    const size_t want = (size_t)POLICY_ENC_IN_DIM + (size_t)POLICY_REF_FF_DIM;
    if (fread(row, sizeof(float), want, fin) != want) {
        return 1;
    }
    memcpy(frame, row, sizeof(float) * POLICY_ENC_IN_DIM);
    memcpy(ref_ff, row + POLICY_ENC_IN_DIM, sizeof(float) * POLICY_REF_FF_DIM);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s frames.bin out.bin\n", argv[0]);
        return 2;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (fin == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    FILE *fout = fopen(argv[2], "wb");
    if (fout == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        fclose(fin);
        return 2;
    }

    // heap: this is a test program, and the struct is intentionally the same one the
    // firmware keeps in BSS. Nothing here imposes on the firmware's constraint.
    policy_state_t *st = calloc(1, sizeof(policy_state_t));
    if (st == NULL) {
        fprintf(stderr, "out of memory\n");
        fclose(fin);
        fclose(fout);
        return 2;
    }
    policy_reset(st);

    float frame[POLICY_ENC_IN_DIM];
    float ref_ff[POLICY_REF_FF_DIM];
    float act[POLICY_ACT_DIM];
    float row[POLICY_ACT_DIM + POLICY_Z_DIM];
    size_t n = 0;

    // INPUT ROWS ARE [frame(33) | ref_ff(3)] = POLICY_ENC_IN_DIM + POLICY_REF_FF_DIM.
    // ref_ff is carried in the same file rather than as a second one so a row is a complete
    // description of one policy step, and because the split it forces here (the frame goes
    // to the encoder, the block goes to the actor) is the exact thing worth checking.
    while (read_step(fin, frame, ref_ff) == 0) {
        policy_step(st, frame, ref_ff, act);
        memcpy(row, act, sizeof(act));
        memcpy(row + POLICY_ACT_DIM, st->z, sizeof(st->z));
        if (fwrite(row, sizeof(float), POLICY_ACT_DIM + POLICY_Z_DIM, fout)
            != POLICY_ACT_DIM + POLICY_Z_DIM) {
            fprintf(stderr, "short write\n");
            break;
        }
        n++;
    }

    fprintf(stderr, "host_check: %zu frames, act_dim %d, z_dim %d, ref_ff_dim %d\n",
            n, POLICY_ACT_DIM, POLICY_Z_DIM, POLICY_REF_FF_DIM);

    free(st);
    fclose(fin);
    fclose(fout);
    return 0;
}
