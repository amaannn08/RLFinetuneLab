from rlfinetunelab.mock.dummy_models import run_local_smoke_test


def test_smoke_pipeline_end_to_end():
    """Runs the in-memory smoke test ensuring zero network calls and full stage integration."""
    result = run_local_smoke_test()
    assert result["status"] == "success"
    assert "train_loss" in result
    assert "eval" in result
    assert "merged_path" in result
