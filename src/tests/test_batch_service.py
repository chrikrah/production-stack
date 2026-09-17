import os
import shutil

import pytest

from vllm_router.services.batch_service import initialize_batch_processor
from vllm_router.services.batch_service.batch import BatchStatus
from vllm_router.services.files_service.file_storage import FileStorage

TEST_BASE_PATH = "/tmp/test_vllm_batch"
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(TEST_BASE_PATH):
        shutil.rmtree(TEST_BASE_PATH)
    yield
    if os.path.exists(TEST_BASE_PATH):
        shutil.rmtree(TEST_BASE_PATH)


def test_initialize_local_batch_processor():
    """app.py calls this whenever --enable-batch-api is passed, so an import
    error inside local_processor aborts router startup."""
    storage = FileStorage(os.path.join(TEST_BASE_PATH, "files"))
    processor = initialize_batch_processor(
        "local", os.path.join(TEST_BASE_PATH, "db"), storage
    )
    assert processor.db_path.endswith("batch_queue.db")


@pytest.mark.asyncio
async def test_create_and_retrieve_batch():
    storage = FileStorage(os.path.join(TEST_BASE_PATH, "files"))
    processor = initialize_batch_processor(
        "local", os.path.join(TEST_BASE_PATH, "db"), storage
    )
    await processor.setup_db()

    created = await processor.create_batch(
        input_file_id="file-1",
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    retrieved = await processor.retrieve_batch(created.id)

    assert retrieved.id == created.id
    assert retrieved.status is BatchStatus.PENDING
    assert retrieved.input_file_id == "file-1"
