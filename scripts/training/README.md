# `training`

Codebase for fine-tuning, packaging, auditing, and evaluating IResNet-100 name learning models across three loss functions (`triplet_margin`, `arcface`, and `supcon`).

## Table of Contents

* [`face_name_metrics`](./face_name_metrics/README.md): Library of utils for exhaustive triplet accuracy evaluation and stratified paired bootstrap confidence intervals.
* [`face_name_metric_learning`](./face_name_metric_learning/README.md): Python package containing utils for data/manifest validation, IResNet-100 loading, training, and embedding export. Consumes utils from [`face_name_metrics`](./face_name_metrics/README.md) for training and validation
* [`notebooks`](./notebooks/README.md): Main notebooks run on [looseleaf](https://looseleafcloud.org/) to convert pretrained weights, run smoke tests, execute the 10 trial GPU fine-tuning process, and export [`artifacts`](../../output/artifacts/README.md) needed for [final evaluation](../eval/README.md). Consumes [`face_name_metric_learning`](./face_name_metric_learning/README.md) for training and eval steps.