# Basic Pytorch template for audio projects

A basic Pytorch template to use as a starting point, with audio projects in mind.

In particular aimed at smaller models that can run on a single GPU, and run in parallel on multi-GPU machines. Each GPU/experiment can have a copy of the entire source code.

## Features

- Simple Dataset classes (included reading list of .wav files from disk, with random cropping).
- On-the-fly mel-spectrogram computation.
- Learning rate schedule, EMA, gradient clipping.
- Checkpoint saving/loading (continuing training).
- Basic training loop for big datasets/slow updates.
