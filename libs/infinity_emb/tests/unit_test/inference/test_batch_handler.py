import asyncio
import concurrent.futures
import copy
import random
import time

import numpy as np
import pytest
import torch

from infinity_emb.inference import BatchHandler
from infinity_emb.inference.models import (
    SentenceTransformerPatched,
)

BATCH_SIZE = 32
N_TIMINGS = 3
LIMIT_SLOWDOWN = 1.25 if torch.cuda.is_available() else 1.35


@pytest.fixture
def temp_threadpool():
    return concurrent.futures.ThreadPoolExecutor()


@pytest.fixture
@pytest.mark.anyio
async def load_patched_bh(temp_threadpool):
    model = SentenceTransformerPatched(pytest.DEFAULT_BERT_MODEL)
    model.encode(["hello " * 512] * BATCH_SIZE)
    bh = BatchHandler(
        model=model, max_batch_size=BATCH_SIZE, threadpool=temp_threadpool
    )
    await bh.spawn()
    return model, bh


@pytest.mark.performance
@pytest.mark.anyio
async def test_batch_performance_raw(get_sts_bechmark_dataset, load_patched_bh):
    model, bh = load_patched_bh
    try:
        model: SentenceTransformerPatched = model
        bh: BatchHandler = bh
        sentences = []
        for d in get_sts_bechmark_dataset:
            for item in d:
                sentences.append(item.texts[0])
        random.shuffle(sentences)
        sentences = sentences if torch.cuda.is_available() else sentences[::4]

        async def method_batch_handler(_sentences):
            _sentences = copy.deepcopy(_sentences)
            start = time.perf_counter()
            _request_size = BATCH_SIZE * 4
            tasks = [
                bh.schedule(
                    _sentences[sl : sl + _request_size],
                )
                for sl in range(0, len(_sentences), _request_size)
            ]
            _ = await asyncio.gather(*tasks)
            end = time.perf_counter()
            return round(end - start, 4)

        def method_patched(_sentences):
            _sentences = copy.deepcopy(_sentences)
            st_s = list(sorted(_sentences))
            start = time.perf_counter()

            emb = []
            for s in range(0, len(_sentences), BATCH_SIZE):
                feat = model.encode_pre(st_s[s : s + BATCH_SIZE])
                embed = model.encode_core(feat)
                emb.append(model.encode_post(embed))
            np.concatenate(emb).tolist()
            end = time.perf_counter()
            return round(end - start, 4)

        def method_st(_sentences):
            _sentences = copy.deepcopy(_sentences)
            start = time.perf_counter()
            _ = model.encode(_sentences, batch_size=BATCH_SIZE).tolist()
            end = time.perf_counter()
            return round(end - start, 4)

        # yappi.get_func_stats().print_all()
        # yappi.stop()
        method_st(sentences[::10])
        time.sleep(0.2)
        time_batch_handler = np.median(
            [(await method_batch_handler(sentences)) for _ in range(N_TIMINGS)]
        )
        time.sleep(0.2)
        time_st_patched = np.median(
            [method_patched(sentences) for _ in range(N_TIMINGS)]
        )
        time.sleep(0.2)
        time_st = np.median([method_st(sentences) for _ in range(N_TIMINGS)])

        print(
            f"times are sentence-transformers: {time_st},"
            " patched-sentence-transformers: "
            f" {time_st_patched}, batch-handler: {time_batch_handler}"
        )

        # WARNING: The timings may depend if you are on debugger.
        # The Python calls may take a significant amount of time.

        assert time_st_patched / time_st < 1.1 if torch.cuda.is_available() else 1.6, (
            "patched-SentenceTransformers slower than"
            f" SentenceTransformers.encode: {time_st_patched} > {time_st}"
        )
        assert time_batch_handler / time_st < LIMIT_SLOWDOWN, (
            "batch_handler slower than Sentence Transformers"
            f" {time_batch_handler}: > {time_st}"
        )
        assert time_batch_handler / time_st_patched < LIMIT_SLOWDOWN, (
            "raw batch_handler threaded queueing looses significant "
            f"time over non-queueing: {time_batch_handler} > {time_st_patched}"
        )

    finally:
        bh.shutdown()
        bh._threadpool.shutdown(wait=True)