#include <stdio.h>
#include <stdlib.h>
#include <rnnoise.h>

#define FRAME_SIZE 80

int main() {
    printf("Testing 8kHz RNNoise...\n");

    // Test frame size
    int frame_size = rnnoise_get_frame_size();
    printf("Frame size: %d (expected: %d)\n", frame_size, FRAME_SIZE);

    // Create RNNoise state
    DenoiseState *st = rnnoise_create(NULL);
    if (st == NULL) {
        printf("Failed to create RNNoise state\n");
        return 1;
    }
    printf("RNNoise state created successfully\n");

    // Generate simple test frame (silence)
    float in[FRAME_SIZE] = {0};
    float out[FRAME_SIZE];

    // Process one frame
    float vad_prob = rnnoise_process_frame(st, out, in);
    printf("VAD probability: %f\n", vad_prob);

    // Cleanup
    rnnoise_destroy(st);
    printf("RNNoise state destroyed\n");

    printf("Full test completed successfully!\n");
    return 0;
}
