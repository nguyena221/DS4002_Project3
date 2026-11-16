# Pilot detector diagnostic

The original 640 x 640 detector input yielded 994 no-face outcomes among 1,000 sampled images. A fixed 20-image diagnostic compared detector input sizes at the same confidence threshold (0.5): 160 without added padding detected one face in 20/20; 320 in 19/20; 640 in 0/20. With a black border of half the maximum source dimension on each side, the corresponding counts were 20, 20, and 19. This supports a detector-scale problem on these tightly cropped inputs; the result does not prove all detections are correct.

The revised pilot uses 160 x 160 with no extra border. The output crop remains 112 x 112, normalized to [-1,1], and the estimated-age cutoff remains 18. The same filenames and image hashes were verified across both pilot runs.

Revised outcomes: 909 retained, 77 estimated under 18, 12 no face, 2 multiple faces. All 909 saved tensors were checked for float32 type, (3,112,112) shape, finite values, and range [-1,1].

Original results remain in OUTPUT/preprocessing_pilot; revised results are in OUTPUT/preprocessing_pilot_160. This is a pilot, not the final screened dataset. Visually review crops and estimates near the age cutoff before a full run. Age estimates do not verify adulthood. MI2 names RetinaFace, while the actual buffalo_l detector is SCRFD; record that implementation difference. No class-size threshold or final splits have been applied.
