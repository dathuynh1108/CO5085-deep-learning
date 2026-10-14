# Working on this repository

This is Huỳnh Thành Đạt's individual CO5085 coursework (2570161, semester 261).
Read `docs/REQUIREMENTS.md` and `docs/GPU_RUNBOOK.md` before changes.

- E1–E3 classify the same Fashion-MNIST images. LSTM/GRU consume image rows or patches, not text.
- E1–E3: no pretrained weights or model zoo backbone. All coursework uses explicit training loops, not high-level Trainer or `trainer.fit`.
- A1 follows `assignments/assignment1-vne.pdf`: torchvision CNN/Transformer and pretrained weights are allowed. Keep A1 code, runs and report separate from E1–E3; A2 waits for the student's topic/paper selection.
- Manual MSA must remain primitive Linear/matmul/softmax/head concatenation. Keep the native reference separately.
- Use validation for checkpoint/model decisions. Explicitly confirm held-out test evaluation only after freezing the protocol.
- Never fabricate measurements, copy reference-repo results or describe an untrained improvement as successful.
- Preserve experiment/source/data hashes. Do not bypass resume guards or mix changed configs into standard reports.
- A1/A2 source PDFs are now in `assignments/`. Use their actual requirements; do not invent A2's selected paper.
- The initial handoff intentionally did not download/train a data batch. A later GPU task may perform full training; do not add a mandatory one-batch preflight.
- Run numerical/protocol unit tests for code changes. Do not claim GPU/end-to-end verification from CPU tensor tests.
- Keep the report practical and moderate in length. Explain mathematics used by the implementation, especially the E3 ablation; don't add unrelated theory.
- Write all reports in neutral Vietnamese academic prose, with minimal personal pronouns (no repeated "tôi"). Present methods and measured findings directly; avoid handoff instructions, third-person advice to the student, integrity lectures, or discussion of supplied templates. In each main report, keep AI disclosure to one sentence linking `AI_USE_REPORT.md`; retain the substantive, accurate disclosure in that separate file. Apply this style to generated analysis as well as LaTeX source.
- Keep `AI_USE_REPORT.md` accurate about assistance, student-supplied decisions and actual checks; do not claim student implementation or measurements without evidence.
