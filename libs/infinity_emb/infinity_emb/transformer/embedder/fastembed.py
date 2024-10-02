import copy
from typing import Dict, List
import numpy as np
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, SparseEmbeddingReturnType, PoolingMethod
from infinity_emb.transformer.abstract import BaseEmbedder
try:
    from fastembed import SparseTextEmbedding
    from fastembed.common.onnx_model import OnnxOutputContext
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
class Fastembed(BaseEmbedder):
    def __init__(self, *, engine_args: EngineArgs) -> None:
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed is not installed." "`pip install infinity-emb[fastembed]`"
            )
        logger.warning(
            "deprecated: fastembed inference"
            " is deprecated and will be removed in the future."
        )
        providers = ["CPUExecutionProvider"]
        if engine_args.device != Device.cpu:
            providers = ["CUDAExecutionProvider"] + providers
        if engine_args.revision is not None:
            logger.warning("revision is not used for fastembed")
        self.model = SparseTextEmbedding(
            model_name=engine_args.model_name_or_path, cache_dir=None, providers=providers
        ).model
        if self.model is None:
            raise ValueError("fastembed model is not available")
        if engine_args.pooling_method != PoolingMethod.auto:
            logger.warning("pooling_method is not used for fastembed")
        self._infinity_tokenizer = copy.deepcopy(self.model.tokenizer)
    def encode_pre(self, sentences: List[str]) -> Dict[str, np.ndarray]:
        encoded = self.model.tokenizer.encode_batch(sentences)
        self.input_ids = np.array([e.ids for e in encoded])
        self.attention_mask = np.array([e.attention_mask for e in encoded])
        onnx_input = {
            "input_ids": np.array(self.input_ids, dtype=np.int64),
            "attention_mask": np.array(self.attention_mask, dtype=np.int64),
            "token_type_ids": np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            ),
        }
        return onnx_input
    def encode_core(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        model_output = self.model.model.run(None, features)
        return model_output[0]
    def encode_post(self, embedding: np.ndarray) -> SparseEmbeddingReturnType:
        token_ids_batch = self.input_ids

        pooled_attention = np.mean(embedding[:, :, 0], axis=1) * self.attention_mask

        for document_token_ids, attention_value in zip(token_ids_batch, pooled_attention):
            document_tokens_with_ids = (
                (idx, self.model.invert_vocab[token_id])
                for idx, token_id in enumerate(document_token_ids)
            )

            reconstructed = self.model._reconstruct_bpe(document_tokens_with_ids)

            filtered = self.model._filter_pair_tokens(reconstructed)

            stemmed = self.model._stem_pair_tokens(filtered)

            weighted = self.model._aggregate_weights(stemmed, attention_value)

            max_token_weight = {}

            for token, weight in weighted:
                max_token_weight[token] = max(max_token_weight.get(token, 0), weight)

            rescored = self.model._rescore_vector(max_token_weight)

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        tks = self._infinity_tokenizer.encode_batch(
            sentences,
        )
        return [len(t.tokens) for t in tks]
