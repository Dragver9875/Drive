from __future__ import annotations


def test_root_import() -> None:
    import drive_kd

    assert drive_kd.__version__ == "0.1.0"


def test_dataset_imports() -> None:
    from drive_kd.datasets import DriveKDDataset, DriveSupervisedDataset, TeacherCacheDataset

    assert DriveSupervisedDataset is not None
    assert TeacherCacheDataset is not None
    assert DriveKDDataset is not None


def test_model_imports() -> None:
    from drive_kd.models import DriveEffB0BiFPNKD, SegFormerB1Teacher

    assert SegFormerB1Teacher is not None
    assert DriveEffB0BiFPNKD is not None


def test_loss_imports() -> None:
    from drive_kd.kd import MultiTaskKDLoss
    from drive_kd.losses import SupervisedMultiTaskLoss

    assert SupervisedMultiTaskLoss is not None
    assert MultiTaskKDLoss is not None


def test_metric_imports() -> None:
    from drive_kd.metrics import DriveEvaluator, benchmark_fps

    assert DriveEvaluator is not None
    assert benchmark_fps is not None


def test_postprocess_imports() -> None:
    from drive_kd.postprocess import DriveThresholds, postprocess_prediction_dict

    assert DriveThresholds is not None
    assert postprocess_prediction_dict is not None