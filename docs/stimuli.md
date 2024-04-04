
## Stimulus generation

### Electrophysiology

The stimuli presented in electrophysiology were originally designed to probe noise invariance; that is, how invariant neural responses to zebra finch song are to the addition of background noise. In this study, only the responses to stimuli in the "clean" condition, with a signal-to-noise ratio of 70 dB, were used.

Scripts from the colony-noise project were used to generate the stimuli used in these experiments. This repository is located at git@gracula.psyc.virginia.edu:pub/smm3rc/colony-noise/. A warning that these are unversioned scripts, and they may not work properly due to bit rot. However, all stimuli used in the study were deposited in neurobank, so there should be no need to re-generate them.

Methods are briefly described here, but examine the README in the colony-noise repository for more information. Files
referenced below are stored in the current repository.

1. Foreground motif selection. Motifs of approximately 2 s in duration were identified from recordings in neurobank. The `inputs/extracellular_songs.yml` file contains references to the recordings by resource name, dataset name, and the selected interval. The `get-songs.py` script was used to extract these segments, resample them to 44.1 kHz, run them through a 150 Hz highpass filter, and scale them to have the same RMS amplitude (-20 dB FS). After inspecting the output, the `get-songs.py` script was run again to deposit the songs into neurobank, assigning them random identifiers.

2. Scene construction. The `inputs/extracellular-scenes.yml` file was generated to define scenes consisting of 10 foreground motifs against a background of synthetic colony noise. The `make-scenes.py` script used this file to construct these scenes, consisting of 10 permutations of the foreground motifs against a background. The amplitude of the foreground was kept constant at -30 dB FS, while the background amplitude varied between -20 dB FS and -100 dB FS. These stimuli were deposited in neurobank, with resource names constructed from the sequence of foreground motifs, the background stimulus, and the amplitudes of foreground and background. The start and stop times of the foreground motifs were stored as metadata in neurobank [we should extract this to a file]
