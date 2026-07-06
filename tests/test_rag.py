"""
tests/test_rag.py

Unit tests for rag/retriever.py's pure-logic pieces (RRF fusion, tokenizer)
and rag/ingestion.py's chunking behavior. These deliberately avoid needing
a real ChromaDB instance, embedding model, or network access — they test
the algorithmic correctness of fusion/chunking directly.

Tests that need a real ChromaDB collection (e.g. full HybridRetriever.retrieve())
are intentionally NOT included here — those are better verified by the
manual smoke test in the README ("Quick sanity test of retrieval alone"),
since spinning up a real embedding model in a unit test is slow and this
project doesn't use a mocking framework heavy enough to fake ChromaDB's
internals meaningfully.

Run with:
    pytest tests/test_rag.py -v
"""

from rag.retriever import HybridRetriever, _simple_tokenize


class TestSimpleTokenize:
    def test_lowercases_and_splits(self):
        assert _simple_tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _simple_tokenize("What's the NHS 111 number?") == ["what", "s", "the", "nhs", "111", "number"]

    def test_handles_empty_string(self):
        assert _simple_tokenize("") == []

    def test_preserves_numbers(self):
        assert _simple_tokenize("Call 999 now") == ["call", "999", "now"]


class TestReciprocalRankFusion:
    def test_top_ranked_in_both_lists_wins(self):
        dense = ["a", "b", "c"]
        bm25 = ["a", "c", "b"]
        fused = HybridRetriever._reciprocal_rank_fusion([dense, bm25])
        assert fused[0] == "a"  # ranked #1 in both lists

    def test_item_only_in_one_list_still_included(self):
        dense = ["a", "b"]
        bm25 = ["c", "d"]
        fused = HybridRetriever._reciprocal_rank_fusion([dense, bm25])
        assert set(fused) == {"a", "b", "c", "d"}

    def test_empty_lists_return_empty(self):
        fused = HybridRetriever._reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_single_list_preserves_relative_order(self):
        # With only one input list, RRF should preserve that list's exact
        # rank order (no other list to perturb the fused score).
        ranked = ["x", "y", "z"]
        fused = HybridRetriever._reciprocal_rank_fusion([ranked])
        assert fused == ranked

    def test_consistently_high_rank_beats_inconsistent_high_rank(self):
        # 'a' is rank 2 in both lists (consistently good).
        # 'b' is rank 1 in list1 but absent from list2 (inconsistent).
        # RRF should still give meaningful credit to consistency — verify
        # 'a' isn't unfairly buried just because it was never literally #1.
        dense = ["b", "a", "c"]
        bm25 = ["a", "c", "d"]
        fused = HybridRetriever._reciprocal_rank_fusion([dense, bm25])
        assert "a" in fused[:2]  # 'a' should be near the top given strong showing in both

    def test_duplicate_ids_across_lists_only_appear_once(self):
        dense = ["a", "b"]
        bm25 = ["a", "b"]
        fused = HybridRetriever._reciprocal_rank_fusion([dense, bm25])
        assert sorted(fused) == ["a", "b"]
        assert len(fused) == 2  # no duplicates


class TestChunkingViaIngester:
    def test_chunker_respects_configured_size_and_overlap(self):
        from rag.ingestion import DocumentIngester
        from config.settings import settings
        from unittest.mock import MagicMock

        # Use a mocked embedding model so this test needs no real
        # sentence-transformers model load (keeps the test fast and
        # dependency-light).
        mock_embedder = MagicMock()
        ingester = DocumentIngester(embedding_model=mock_embedder)

        long_text = "This is a sentence. " * 100  # ~2000 chars
        chunks = ingester.splitter.split_text(long_text)

        assert len(chunks) > 1  # text longer than chunk_size must be split
        for chunk in chunks:
            # Allow some slack over the configured chunk_size since the
            # splitter prefers breaking on sentence boundaries over a hard
            # mid-word cut. We check against the public settings value
            # (not any private langchain internal) to keep this test
            # resilient to langchain version changes.
            assert len(chunk) <= settings.chunk_size + 100

    def test_chunks_have_overlap(self):
        from rag.ingestion import DocumentIngester
        from unittest.mock import MagicMock

        mock_embedder = MagicMock()
        ingester = DocumentIngester(embedding_model=mock_embedder)

        # Build a long text out of distinct, searchable sentence markers so
        # we can directly check whether the tail of one chunk reappears at
        # the head of the next — that's what "overlap" actually means here.
        long_text = " ".join(f"MARKER{i:03d} filler words to add length here." for i in range(60))
        chunks = ingester.splitter.split_text(long_text)

        assert len(chunks) > 1, "Test text wasn't long enough to force a split — check chunk_size config"

        overlap_found = False
        for i in range(len(chunks) - 1):
            tail = chunks[i][-30:]
            if tail.strip() and tail.strip() in chunks[i + 1]:
                overlap_found = True
                break

        assert overlap_found, "Expected at least one pair of consecutive chunks to share overlapping text"
