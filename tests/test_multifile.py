"""Tests for multi-file ingestion: discovery, labelling, append/replace/skip/remove, per-source filtering."""
import _isolation  # noqa: F401  (must be first: keeps tests out of <project>/.raglogger)
import gzip
import io
import os
import shutil
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from raglogger.chunker import LogChunker
from raglogger.index import Bm25Index
from raglogger.models import SearchFilters
from raglogger.parser import ConsoleLogParser
from raglogger.redaction import Redactor
from raglogger.service import SOURCES_METADATA_KEY, IngestService
from raglogger.sources import assign_labels, discover

FIXTURE = Path(__file__).parent / "fixtures" / "sample-console.log"
NODE2_LINE = ("INFO   | jvm 1    | main    | 2026/08/30 09:00:00.000 | [2026-08-30T09:00:00,000] ERROR [hybrisHTTP2] "
              "[OrderExportJob] Order export to SAP failed for order 4711: Connection reset\n")


class _RecordingVectorStore:
    """Test double for VectorStoreService that records full rebuilds vs incremental upserts."""

    def __init__(self):
        self.calls = []

    def build(self, chunks):
        self.calls.append(("build", len(chunks), [], []))
        return len(chunks)

    def upsert_sources(self, chunks, replace_sources, remove_sources=None):
        self.calls.append(("upsert", len(chunks), list(replace_sources), list(remove_sources or [])))
        return len(chunks)


class MultiFileTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for node in ("node1", "node2"):
            (self.root / node).mkdir()
            shutil.copy(FIXTURE, self.root / node / "console.log")
        with open(self.root / "node2" / "console.log", "a", encoding="utf-8") as handle:
            handle.write(NODE2_LINE)
        self.index_path = self.root / "idx.json.gz"
        self.vectors = _RecordingVectorStore()

    def tearDown(self):
        self.tmp.cleanup()

    def service(self, vectors=True):
        return IngestService(ConsoleLogParser(), Redactor(), LogChunker(), self.vectors if vectors else None)

    def load(self):
        return Bm25Index.load(self.index_path)

    def labels(self):
        return sorted(self.load().metadata[SOURCES_METADATA_KEY])

class DiscoveryAndLabelTest(MultiFileTestBase):
    def test_directory_is_searched_recursively_and_deduplicated(self):
        files = discover([self.root, self.root / "node1" / "console.log"])
        expected = [self.root.resolve() / "node1" / "console.log", self.root.resolve() / "node2" / "console.log"]
        self.assertEqual(expected, files)

    def test_missing_path_and_empty_directory(self):
        with self.assertRaisesRegex(FileNotFoundError, "Console log not found"):
            discover([self.root / "nope.log"])
        (self.root / "empty").mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "No files matching"):
            discover([self.root / "empty"])

    def test_clashing_file_names_get_parent_directory(self):
        files = assign_labels(discover([self.root]))
        self.assertEqual(["node1/console.log", "node2/console.log"], [item.label for item in files])
        self.assertNotEqual(files[0].key, files[1].key)

    def test_unique_names_stay_short_and_existing_path_keeps_its_label(self):
        other = self.root / "console-20260830.log"
        shutil.copy(FIXTURE, other)
        self.assertEqual(["console-20260830.log"], [item.label for item in assign_labels([other.resolve()])])
        node1 = (self.root / "node1" / "console.log").resolve()
        self.assertEqual(["node1/console.log"],
                         [item.label for item in assign_labels([node1], {"node1/console.log": str(node1)})])

    def test_new_file_with_same_name_as_indexed_one_gets_distinct_label(self):
        node2 = (self.root / "node2" / "console.log").resolve()
        existing = {"console.log": str((self.root / "node1" / "console.log").resolve())}
        self.assertEqual(["node2/console.log"], [item.label for item in assign_labels([node2], existing)])

    def test_explicit_label_rules(self):
        path = (self.root / "node1" / "console.log").resolve()
        self.assertEqual("app01", assign_labels([path], explicit_label="app01")[0].label)
        with self.assertRaisesRegex(ValueError, "exactly one file"):
            assign_labels(discover([self.root]), explicit_label="x")
        with self.assertRaisesRegex(ValueError, "already used"):
            assign_labels([path], {"app01": "/some/other.log"}, explicit_label="app01")

class IngestManyTest(MultiFileTestBase):
    def test_rebuild_with_two_files_keeps_both_sources_separate(self):
        stats = self.service().ingest_many([self.root], self.index_path)
        self.assertEqual(2, stats.total_sources)
        index = self.load()
        self.assertEqual(["node1/console.log", "node2/console.log"], self.labels())
        self.assertEqual(len(index.chunks), len({chunk.chunk_id for chunk in index.chunks}), "chunk ids collide")
        self.assertEqual({"node1/console.log", "node2/console.log"}, {chunk.source for chunk in index.chunks})
        self.assertEqual(("build", stats.total_chunks, [], []), self.vectors.calls[-1])

    def test_identical_problem_in_two_files_appears_once_per_source(self):
        self.service().ingest_many([self.root], self.index_path)
        hits = self.load().search("IllegalArgumentException removing item", top_k=10,
                                  filters=SearchFilters(kind="incident"))
        self.assertEqual({"node1/console.log", "node2/console.log"}, {hit.chunk.source for hit in hits})

    def test_source_filter_and_node_specific_error(self):
        self.service().ingest_many([self.root], self.index_path)
        index = self.load()
        self.assertEqual("node2/console.log", index.search("order export SAP failed", top_k=3)[0].chunk.source)
        only_node1 = index.search("order export SAP failed", top_k=10, filters=SearchFilters(source="NODE1"))
        self.assertTrue(all(hit.chunk.source == "node1/console.log" for hit in only_node1))
        self.assertFalse(any("OrderExportJob" in hit.chunk.title for hit in only_node1))

    def test_append_adds_new_file_and_skips_unchanged(self):
        self.service().ingest_many([self.root / "node1"], self.index_path)
        self.assertEqual(["console.log"], self.labels())
        stats = self.service().ingest_many([self.root], self.index_path, append=True)
        self.assertEqual(["console.log", "node2/console.log"], self.labels())
        self.assertEqual({"console.log": True, "node2/console.log": False},
                         {item.label: item.skipped for item in stats.files})
        self.assertEqual("upsert", self.vectors.calls[-1][0])
        self.assertEqual([], self.vectors.calls[-1][2], "nothing replaced")

    def test_append_replaces_changed_file_without_duplicates(self):
        self.service().ingest_many([self.root], self.index_path)
        before = len(self.load().chunks)
        log = self.root / "node1" / "console.log"
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(NODE2_LINE.replace("OrderExportJob", "PaymentCaptureJob"))
        os.utime(log, (time.time() + 5, time.time() + 5))
        stats = self.service().ingest_many([log], self.index_path, append=True)
        index = self.load()
        self.assertEqual(len(index.chunks), len({chunk.chunk_id for chunk in index.chunks}))
        self.assertGreater(len(index.chunks), before)
        self.assertEqual(["node1/console.log"], self.vectors.calls[-1][2])
        self.assertFalse(stats.files[0].skipped)
        self.assertEqual("node1/console.log", index.search("PaymentCaptureJob", top_k=1)[0].chunk.source)

    def test_force_reingests_unchanged_file(self):
        self.service().ingest_many([self.root], self.index_path)
        stats = self.service().ingest_many([self.root / "node1"], self.index_path, append=True, force=True)
        self.assertFalse(stats.files[0].skipped)
        self.assertEqual(2, stats.total_sources)

    def test_remove_source(self):
        self.service().ingest_many([self.root], self.index_path)
        stats = self.service().ingest_many([], self.index_path, remove=["node2/console.log"])
        self.assertEqual(["node1/console.log"], self.labels())
        self.assertTrue(all(chunk.source == "node1/console.log" for chunk in self.load().chunks))
        self.assertEqual(("upsert", 0, [], ["node2/console.log"]), self.vectors.calls[-1])
        self.assertEqual(["node2/console.log"], stats.removed_sources)
        with self.assertRaisesRegex(ValueError, "unknown source"):
            self.service().ingest_many([], self.index_path, remove=["node9"])

    def test_rebuild_drops_sources_not_listed(self):
        self.service().ingest_many([self.root], self.index_path)
        self.service().ingest_many([self.root / "node2"], self.index_path)
        self.assertEqual(["console.log"], self.labels())

    def test_gzip_rotated_log(self):
        rotated = self.root / "console.log.1.gz"
        with gzip.open(rotated, "wt", encoding="utf-8") as handle:
            handle.write(FIXTURE.read_text(encoding="utf-8"))
        stats = self.service(vectors=False).ingest_many([rotated], self.index_path)
        self.assertEqual(13, stats.lines)
        self.assertEqual(["console.log.1.gz"], self.labels())

    def test_legacy_index_cannot_be_appended(self):
        legacy = Bm25Index()
        legacy.build([], metadata={"source": "old.log"})
        legacy.save(self.index_path)
        with self.assertRaisesRegex(ValueError, "older single-file version"):
            self.service().ingest_many([self.root], self.index_path, append=True)

class CliMultiFileTest(MultiFileTestBase):
    def run_cli(self, *argv):
        from raglogger.cli import main
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--env-file", "off", "--engine", "native", "--index", str(self.index_path), *argv])
        return code, out.getvalue(), err.getvalue()

    def test_ingest_two_files_list_sources_and_filter(self):
        code, out, err = self.run_cli("ingest", str(self.root / "node1" / "console.log"),
                                      str(self.root / "node2" / "console.log"))
        self.assertEqual(0, code, err)
        self.assertIn("+ node1/console.log", out)
        self.assertIn("2 source(s)", out)
        code, out, _ = self.run_cli("sources")
        self.assertEqual(0, code)
        self.assertIn("node1/console.log", out)
        self.assertIn("node2/console.log", out)
        code, out, _ = self.run_cli("search", "order", "export", "--source", "node2")
        self.assertIn("[node2/console.log]", out)
        self.assertNotIn("[node1/console.log]", out)

    def test_append_skip_and_remove_messages(self):
        self.run_cli("ingest", str(self.root))
        code, out, _ = self.run_cli("ingest", "--append", str(self.root))
        self.assertEqual(0, code)
        self.assertEqual(2, out.count("unchanged, skipped"))
        code, out, _ = self.run_cli("ingest", "--remove", "node1/console.log")
        self.assertEqual(0, code)
        self.assertIn("- node1/console.log: removed", out)
        self.assertIn("1 source(s)", out)

    def test_errors_for_unknown_source_filter_and_empty_ingest(self):
        self.run_cli("ingest", str(self.root))
        code, _, err = self.run_cli("search", "x", "--source", "node9")
        self.assertEqual(2, code)
        self.assertIn("Indexed sources: node1/console.log, node2/console.log", err)
        code, _, err = self.run_cli("ingest")
        self.assertEqual(2, code)
        self.assertIn("Nothing to ingest", err)

    def test_source_filter_matches_original_path_when_label_is_short(self):
        # Unique file names keep short labels; --source must still find them via their directory.
        shutil.move(str(self.root / "node2" / "console.log"), str(self.root / "node2" / "console-node.log"))
        self.run_cli("ingest", str(self.root))
        code, out, _ = self.run_cli("sources")
        self.assertIn("console-node.log", out)
        code, out, err = self.run_cli("search", "order", "export", "--source", "NODE2")
        self.assertEqual(0, code, err)
        self.assertIn("[console-node.log]", out)
        self.assertNotIn("[console.log]", out)


try:
    import langchain_core  # noqa: F401
    HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    HAS_LANGCHAIN = False


@unittest.skipUnless(HAS_LANGCHAIN, "LangChain requirements not installed")
class ChromaIncrementalTest(MultiFileTestBase):
    """Real Chroma round-trip: incremental updates must only touch the affected source's vectors."""

    def setUp(self):
        super().setUp()
        from tests.test_langchain_rag import KeywordHashEmbedding
        from raglogger.vectorstore import EmbeddingSettings, VectorStoreService
        self.store = VectorStoreService(self.root / "chroma", KeywordHashEmbedding(),
                                        EmbeddingSettings(provider="local", model="fake"))

    def sources_in_store(self):
        metadatas = self.store.open().get(include=["metadatas"])["metadatas"]
        counts = {}
        for meta in metadatas:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return counts

    def test_append_replace_and_remove_keep_vectors_in_sync(self):
        service = IngestService(ConsoleLogParser(), Redactor(), LogChunker(), self.store)
        service.ingest_many([self.root / "node1"], self.index_path)
        self.assertEqual(["console.log"], list(self.sources_in_store()))
        service.ingest_many([self.root / "node2"], self.index_path, append=True)
        counts = self.sources_in_store()
        self.assertEqual({"console.log", "node2/console.log"}, set(counts))
        service.ingest_many([self.root / "node2"], self.index_path, append=True, force=True)
        self.assertEqual(counts, self.sources_in_store(), "replace must not duplicate vectors")
        service.ingest_many([], self.index_path, remove=["console.log"])
        self.assertEqual({"node2/console.log"}, set(self.sources_in_store()))
        bm25_ids = {chunk.chunk_id for chunk in self.load().chunks}
        self.assertEqual(bm25_ids, set(self.store.open().get()["ids"]), "BM25 index and vectors diverged")


if __name__ == "__main__":
    unittest.main()
