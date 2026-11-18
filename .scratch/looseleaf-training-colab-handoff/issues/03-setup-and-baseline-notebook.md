# Setup and baseline export

Status: ready-for-agent
Type: task
Blocked by: 02

Create the setup/smoke-test notebook and baseline embedding export notebook.
Their behavior must be externally verifiable through generated environment
records and a valid baseline NPZ.

Acceptance:

- Setup notebook records runtime/package/GPU details and validates prepared data.
- Pretrained IResNet-100 smoke test produces one normalized 512-D embedding.
- Baseline export embeds validation/test without augmentation.
- Baseline NPZ matches the artifact contract and uses stable h5 indices.
- Notebooks have cleared outputs, explicit paths, one configuration section,
  and fail-fast assertions.

References:

- [spec](../spec.md)
- [handoff design](../../../docs/face-name-metric-learning-handoff.md)

