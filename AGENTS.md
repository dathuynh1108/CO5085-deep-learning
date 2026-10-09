# Working on this repository

This is Huỳnh Thành Đạt's individual CO5085 coursework (2570161, semester 261).
Read `docs/REQUIREMENTS.md` and `docs/GPU_RUNBOOK.md` before changes.

- E1–E3 classify the same Fashion-MNIST images. LSTM/GRU consume image rows or patches, not text.
- No pretrained weights, model zoo backbone, high-level Trainer or `trainer.fit`.
- Manual MSA must remain primitive Linear/matmul/softmax/head concatenation. Keep the native reference separately.
- Use validation for checkpoint/model decisions. Explicitly confirm held-out test evaluation only after freezing the protocol.
- Never fabricate measurements, copy reference-repo results or describe an untrained improvement as successful.
- Preserve experiment/source/data hashes. Do not bypass resume guards or mix changed configs into standard reports.
- A1/A2 instructions are missing. Do not invent their scope.
- The initial handoff intentionally did not download/train a data batch. A later GPU task may perform full training; do not add a mandatory one-batch preflight.
- Run numerical/protocol unit tests for code changes. Do not claim GPU/end-to-end verification from CPU tensor tests.
- Keep the report practical and moderate in length. Explain mathematics used by the implementation, especially the E3 ablation; don't add unrelated theory.
- Initial code/report is substantially AI-assisted and not ready for submission under the course's AI-use restrictions without the student's own work and instructor clarification.
