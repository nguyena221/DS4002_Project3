# Do You Look Like Your Name?

DS4002 Project 3 | 
Authors: Eli Boyden, Zilan Saadi, Annie Nguyen | 
Prof. Loreto Alonzi

## Summary

This project has three main stages: preprocessing, analysis and evaluation. During preprocessing we filter out unidentifiable faces, and faces with an estimated age < 25, and align/crop images. During analysis we fine tune a IResNet-100 model to predict names based on face image data, and produce cosine simularity data from the image embeddings. During evaluation we compare our fine tuned models against a non-fine-tuned IResNet-100 baseline, targeting a statistically significant improvement in simularity lift.
