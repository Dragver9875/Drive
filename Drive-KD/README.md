# Drive-KD

**Drive-KD** is a standalone knowledge-distillation pipeline for road, lane, and edge segmentation.

```text
SegFormer-B1 Teacher
        ↓
Offline teacher cache generation
        ↓
Drive-EffB0-BiFPN-KD Student
        ↓
Evaluation + export