// One-frame probe: dump every intermediate of the deployed forward pass.
//
// `policy_host_check.py` reports that C and torch disagree; this narrows it to a STAGE
// (standardization / gate pre-activations / hidden state / GELU / latent) by writing the
// intermediates of a single frame to a file that scratch/probe_policy_net_c.py diffs
// against the same quantities computed in numpy.
//
//   usage: net_probe frame.bin out.bin
//     frame.bin : ONE row of (33 + 3) float32 = [raw frame | ref feed-forward] - the same
//                 row format host_check.c reads, so a failing row can be fed straight here.

#include <stdio.h>
#include <string.h>

#include "policy_net.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s frame.bin out.bin\n", argv[0]);
        return 2;
    }
    FILE *fin = fopen(argv[1], "rb");
    FILE *fout = fopen(argv[2], "wb");
    if (fin == NULL || fout == NULL) {
        return 2;
    }

    static policy_state_t st;
    policy_reset(&st);

    float frame[POLICY_ENC_IN_DIM];
    float ref_ff[POLICY_REF_FF_DIM];
    float row[POLICY_ENC_IN_DIM + POLICY_REF_FF_DIM];
    if (fread(row, sizeof(float), POLICY_ENC_IN_DIM + POLICY_REF_FF_DIM, fin)
        != (size_t)POLICY_ENC_IN_DIM + POLICY_REF_FF_DIM) {
        fclose(fin);
        fclose(fout);
        return 2;
    }
    memcpy(frame, row, sizeof(float) * POLICY_ENC_IN_DIM);
    memcpy(ref_ff, row + POLICY_ENC_IN_DIM, sizeof(float) * POLICY_REF_FF_DIM);
    (void)ref_ff;   // the encoder stages below do not consume it; kept for the row contract

    policy_standardize(frame, st.x_std);
    fwrite(st.x_std, sizeof(float), POLICY_ENC_IN_DIM, fout);              // stage 1: 33

    policy_encoder_step(&st, frame);
    for (int i = 0; i < 3 * POLICY_GRU_HIDDEN; i++) {
        st.gi[i] += st.gh[i];                                              // total gate pre-activation
    }
    fwrite(st.gi, sizeof(float), 3 * POLICY_GRU_HIDDEN, fout);             // stage 2: 144
    fwrite(st.h, sizeof(float), POLICY_GRU_HIDDEN, fout);                  // stage 3: 48
    fwrite(st.head_a, sizeof(float), POLICY_GRU_HIDDEN, fout);             // stage 4: 48 (post-GELU)
    fwrite(st.z, sizeof(float), POLICY_Z_DIM, fout);                       // stage 5: 16

    fclose(fin);
    fclose(fout);
    return 0;
}
