"""Tests for ETL orchestration component."""

import argparse
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.etl.orchestrator import (
    ComponentResult,
    ComponentStatus,
    ETLOrchestrator,
    OrchestrationConfig,
    create_argument_parser,
    main,
)


class TestComponentStatus:
    """Test ComponentStatus enum."""

    def test_component_status_values(self):
        """Test that ComponentStatus has expected values."""
        assert ComponentStatus.PENDING.value == "pending"
        assert ComponentStatus.RUNNING.value == "running"
        assert ComponentStatus.COMPLETED.value == "completed"
        assert ComponentStatus.FAILED.value == "failed"
        assert ComponentStatus.SKIPPED.value == "skipped"


class TestComponentResult:
    """Test ComponentResult dataclass."""

    def test_component_result_creation(self):
        """Test basic ComponentResult creation."""
        result = ComponentResult(
            component_name="test_component",
            status=ComponentStatus.PENDING,
            start_time=None,
            end_time=None,
            error=None,
            metrics={},
            output_files=[],
        )
        assert result.component_name == "test_component"
        assert result.status == ComponentStatus.PENDING
        assert result.start_time is None
        assert result.end_time is None
        assert result.error is None
        assert result.metrics == {}
        assert result.output_files == []

    def test_component_result_duration_calculation(self):
        """Test duration property calculation."""
        # Test with both start and end times
        result = ComponentResult(
            component_name="test",
            status=ComponentStatus.COMPLETED,
            start_time=100.0,
            end_time=150.0,
            error=None,
            metrics={},
            output_files=[],
        )
        assert result.duration == 50.0

        # Test with missing start time
        result.start_time = None
        assert result.duration is None

        # Test with missing end time
        result.start_time = 100.0
        result.end_time = None
        assert result.duration is None

    def test_component_result_with_error(self):
        """Test ComponentResult with error information."""
        result = ComponentResult(
            component_name="failed_component",
            status=ComponentStatus.FAILED,
            start_time=100.0,
            end_time=120.0,
            error="Test error message",
            metrics={"attempts": 3},
            output_files=[],
        )
        assert result.status == ComponentStatus.FAILED
        assert result.error == "Test error message"
        assert result.metrics["attempts"] == 3

    def test_component_result_with_output_files(self):
        """Test ComponentResult with output files."""
        output_files = [Path("/tmp/file1.txt"), Path("/tmp/file2.txt")]
        result = ComponentResult(
            component_name="processor",
            status=ComponentStatus.COMPLETED,
            start_time=100.0,
            end_time=120.0,
            error=None,
            metrics={"processed_files": 2},
            output_files=output_files,
        )
        assert len(result.output_files) == 2
        assert result.output_files[0] == Path("/tmp/file1.txt")
        assert result.output_files[1] == Path("/tmp/file2.txt")


class TestOrchestrationConfig:
    """Test OrchestrationConfig dataclass."""

    def test_orchestration_config_defaults(self):
        """Test OrchestrationConfig with default values."""
        config = OrchestrationConfig(
            config_path=Path("/tmp/config.yaml"),
        )
        assert config.config_path == Path("/tmp/config.yaml")
        assert config.storage_type is None
        assert config.log_level == "INFO"
        assert config.dry_run is False
        assert config.skip_scraping is False
        assert config.scraper_concurrency == 2
        assert config.scraper_delay == 2.0
        assert config.scraper_limit is None
        assert config.download_concurrency == 3
        assert config.download_limit is None
        assert config.overwrite_files is False
        assert config.min_file_size == 1024
        assert config.max_file_size == 100_000_000
        assert config.force_reprocess is False
        assert config.ocr_language == ["eng"]
        assert config.extract_images is False
        assert config.min_chars_per_page == 100
        assert config.transcripts_input is None
        assert config.transcripts_limit is None
        assert config.max_tokens is None

    def test_orchestration_config_custom_values(self):
        """Test OrchestrationConfig with custom values."""
        config = OrchestrationConfig(
            config_path=Path("/custom/config.yaml"),
            storage_type="cloud",
            log_level="DEBUG",
            dry_run=True,
            skip_scraping=True,
            scraper_concurrency=5,
            scraper_delay=1.5,
            scraper_limit=20,
            download_concurrency=10,
            download_limit=50,
            overwrite_files=True,
            min_file_size=2048,
            max_file_size=50_000_000,
            force_reprocess=True,
            ocr_language=["eng", "spa"],
            extract_images=True,
            min_chars_per_page=200,
            transcripts_input=Path("/transcripts"),
            transcripts_limit=25,
            max_tokens=4000,
        )
        assert config.storage_type == "cloud"
        assert config.log_level == "DEBUG"
        assert config.dry_run is True
        assert config.skip_scraping is True
        assert config.scraper_concurrency == 5
        assert config.scraper_delay == 1.5
        assert config.scraper_limit == 20
        assert config.download_concurrency == 10
        assert config.download_limit == 50
        assert config.overwrite_files is True
        assert config.min_file_size == 2048
        assert config.max_file_size == 50_000_000
        assert config.force_reprocess is True
        assert config.ocr_language == ["eng", "spa"]
        assert config.extract_images is True
        assert config.min_chars_per_page == 200
        assert config.transcripts_input == Path("/transcripts")
        assert config.transcripts_limit == 25
        assert config.max_tokens == 4000


class TestCreateArgumentParser:
    """Test create_argument_parser function."""

    def test_argument_parser_creation(self):
        """Test that argument parser is created with expected arguments."""
        parser = create_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

        # Test parsing with default values
        args = parser.parse_args([])
        assert str(args.config) == "backend/etl/config.yaml"
        assert args.storage_type is None
        assert args.log_level == "INFO"
        assert args.dry_run is False
        assert args.skip_scraping is False

    def test_argument_parser_with_custom_values(self):
        """Test argument parser with custom command-line arguments."""
        parser = create_argument_parser()

        args = parser.parse_args(
            [
                "--config",
                "custom_config.yaml",
                "--storage-type",
                "cloud",
                "--log-level",
                "DEBUG",
                "--dry-run",
                "--skip-scraping",
                "--scraper-concurrency",
                "5",
                "--download-concurrency",
                "10",
                "--download-limit",
                "100",
                "--overwrite-files",
                "--force-reprocess",
                "--extract-images",
                "--transcripts-limit",
                "50",
                "--max-tokens",
                "5000",
            ]
        )

        assert str(args.config) == "custom_config.yaml"
        assert args.storage_type == "cloud"
        assert args.log_level == "DEBUG"
        assert args.dry_run is True
        assert args.skip_scraping is True
        assert args.scraper_concurrency == 5
        assert args.download_concurrency == 10
        assert args.download_limit == 100
        assert args.overwrite_files is True
        assert args.force_reprocess is True
        assert args.extract_images is True
        assert args.transcripts_limit == 50
        assert args.max_tokens == 5000


class TestETLOrchestrator:
    """Test ETLOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test ETLOrchestrator initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(config_path=config_path)

            with patch("backend.etl.orchestrator.logger") as mock_logger:
                orchestrator = ETLOrchestrator(config)
                assert orchestrator.config == config
                mock_logger.remove.assert_called_once()
                mock_logger.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_component_success(self):
        """Test successful component execution."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(config_path=config_path)

            with patch("backend.etl.orchestrator.logger"):
                orchestrator = ETLOrchestrator(config)

                # Mock a successful component function
                async def mock_component(result):
                    result.metrics = {"processed": 10}
                    result.output_files = [Path("/tmp/output.txt")]
                    await asyncio.sleep(0.1)  # Simulate processing time

                result = await orchestrator._execute_component(
                    "test_component", mock_component
                )

                assert result.component_name == "test_component"
                assert result.status == ComponentStatus.COMPLETED
                assert result.start_time is not None
                assert result.end_time is not None
                assert result.duration is not None
                assert result.duration >= 0.1
                assert result.error is None
                assert result.metrics == {"processed": 10}
                assert result.output_files == [Path("/tmp/output.txt")]

    @pytest.mark.asyncio
    async def test_execute_component_failure(self):
        """Test component execution with failure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(config_path=config_path)

            with patch("backend.etl.orchestrator.logger"):
                orchestrator = ETLOrchestrator(config)

                # Mock a failing component function
                async def mock_failing_component(result):
                    raise RuntimeError("Component failed")

                result = await orchestrator._execute_component(
                    "failing_component", mock_failing_component
                )

                assert result.component_name == "failing_component"
                assert result.status == ComponentStatus.FAILED
                assert result.start_time is not None
                assert result.end_time is not None
                assert result.duration is not None
                assert "Component failed" in result.error
                assert result.metrics == {}
                assert result.output_files == []

    @pytest.mark.asyncio
    async def test_simulate_pipeline(self):
        """Test pipeline simulation for dry run."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(config_path=config_path, dry_run=True)

            with patch("backend.etl.orchestrator.logger"):
                orchestrator = ETLOrchestrator(config)

                results = await orchestrator._simulate_pipeline()

                assert len(results) == 6  # Six components (added Embeddings Generator)
                component_names = [r.component_name for r in results]
                assert "Growth Lab Scraper" in component_names
                assert "Growth Lab File Downloader" in component_names
                assert "PDF Processor" in component_names
                assert "Lecture Transcripts Processor" in component_names
                assert "Text Chunker" in component_names
                assert "Embeddings Generator" in component_names

                # All components should be completed in simulation
                for result in results:
                    assert result.status == ComponentStatus.COMPLETED
                    assert result.start_time is not None
                    assert result.end_time is not None
                    assert result.duration is not None
                    assert result.error is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_pipeline_dry_run(self):
        """Integration test for dry run pipeline execution."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(config_path=config_path, dry_run=True)

            with patch("backend.etl.orchestrator.logger"):
                orchestrator = ETLOrchestrator(config)

                results = await orchestrator.run_pipeline()

                assert len(results) == 6  # Six components
                for result in results:
                    assert result.status == ComponentStatus.COMPLETED
                    assert result.duration is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_pipeline_with_mocked_components(self):
        """Integration test with mocked components."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(
                config_path=config_path,
                skip_scraping=False,
                dry_run=False,
            )

            with patch("backend.etl.orchestrator.logger"):
                orchestrator = ETLOrchestrator(config)

                # Mock all component methods
                with (
                    patch.object(orchestrator, "_run_scraper") as mock_scraper,
                    patch.object(
                        orchestrator, "_run_file_downloader"
                    ) as mock_downloader,
                    patch.object(orchestrator, "_run_pdf_processor") as mock_processor,
                    patch.object(
                        orchestrator, "_run_lecture_transcripts"
                    ) as mock_transcripts,
                    patch.object(orchestrator, "_run_text_chunker") as mock_chunker,
                ):
                    # Configure mocks to simulate successful execution
                    async def mock_component_success(result):
                        result.metrics = {"processed": 5}
                        result.output_files = [Path("/tmp/test_output.txt")]

                    mock_scraper.side_effect = mock_component_success
                    mock_downloader.side_effect = mock_component_success
                    mock_processor.side_effect = mock_component_success
                    mock_transcripts.side_effect = mock_component_success
                    mock_chunker.side_effect = mock_component_success

                    results = await orchestrator.run_pipeline()

                    assert len(results) == 6  # Six components
                    # Most components should succeed
                    # (embeddings generator may be skipped)
                    for result in results:
                        if result.component_name != "Embeddings Generator":
                            assert result.status == ComponentStatus.COMPLETED
                            assert result.metrics == {"processed": 5}
                            assert result.output_files == [Path("/tmp/test_output.txt")]

                    # Verify components were called in correct order
                    mock_scraper.assert_called_once()
                    mock_downloader.assert_called_once()
                    mock_processor.assert_called_once()
                    mock_transcripts.assert_called_once()
                    mock_chunker.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_pipeline_with_component_failure(self):
        """Integration test with component failure handling."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(config_path=config_path, dry_run=False)

            with patch("backend.etl.orchestrator.logger"):
                orchestrator = ETLOrchestrator(config)

                # Mock components with one failure
                with (
                    patch.object(orchestrator, "_run_scraper") as mock_scraper,
                    patch.object(
                        orchestrator, "_run_file_downloader"
                    ) as mock_downloader,
                    patch.object(orchestrator, "_run_pdf_processor") as mock_processor,
                    patch.object(
                        orchestrator, "_run_lecture_transcripts"
                    ) as mock_transcripts,
                    patch.object(orchestrator, "_run_text_chunker") as mock_chunker,
                ):
                    # First component succeeds
                    async def mock_success(result):
                        result.metrics = {"processed": 3}
                        result.output_files = [Path("/tmp/success.txt")]

                    # Second component fails
                    async def mock_failure(result):
                        raise RuntimeError("Simulated component failure")

                    mock_scraper.side_effect = mock_success
                    mock_downloader.side_effect = mock_failure
                    mock_processor.side_effect = mock_success
                    mock_transcripts.side_effect = mock_success
                    mock_chunker.side_effect = mock_success

                    results = await orchestrator.run_pipeline()

                    assert len(results) == 6  # Six components

                    # First component should succeed
                    assert results[0].status == ComponentStatus.COMPLETED
                    assert results[0].component_name == "Growth Lab Scraper"

                    # Second component should fail
                    assert results[1].status == ComponentStatus.FAILED
                    assert results[1].component_name == "Growth Lab File Downloader"
                    assert "Simulated component failure" in results[1].error

                    # Remaining components should still run
                    assert results[2].status == ComponentStatus.COMPLETED
                    assert results[3].status == ComponentStatus.COMPLETED
                    assert results[4].status == ComponentStatus.COMPLETED
                    # Embeddings generator may be skipped if no chunks found
                    assert results[5].component_name == "Embeddings Generator"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_pipeline_skip_scraping(self):
        """Integration test with scraping skipped."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            config = OrchestrationConfig(
                config_path=config_path,
                skip_scraping=True,
                dry_run=False,
            )

            with patch("backend.etl.orchestrator.logger"):
                orchestrator = ETLOrchestrator(config)

                # Mock components (except scraper, which handles skip_scraping itself)
                with (
                    patch.object(
                        orchestrator, "_run_file_downloader"
                    ) as mock_downloader,
                    patch.object(orchestrator, "_run_pdf_processor") as mock_processor,
                    patch.object(
                        orchestrator, "_run_lecture_transcripts"
                    ) as mock_transcripts,
                    patch.object(orchestrator, "_run_text_chunker") as mock_chunker,
                ):

                    async def mock_component_success(result):
                        result.metrics = {"processed": 2}
                        result.output_files = [Path("/tmp/output.txt")]

                    mock_downloader.side_effect = mock_component_success
                    mock_processor.side_effect = mock_component_success
                    mock_transcripts.side_effect = mock_component_success
                    mock_chunker.side_effect = mock_component_success

                    results = await orchestrator.run_pipeline()

                    assert len(results) == 6  # Six components

                    # First component should be skipped
                    assert results[0].status == ComponentStatus.SKIPPED
                    assert results[0].component_name == "Growth Lab Scraper"

                    # Other components should succeed (embeddings may be skipped)
                    for i in range(1, 6):
                        if results[i].component_name != "Embeddings Generator":
                            assert results[i].status == ComponentStatus.COMPLETED


class TestMainFunction:
    """Test main function and CLI integration."""

    @pytest.mark.asyncio
    async def test_main_function_dry_run(self):
        """Test main function with dry run."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            # Mock command line arguments
            test_args = [
                "orchestrator",
                "--config",
                str(config_path),
                "--dry-run",
                "--log-level",
                "DEBUG",
            ]

            with (
                patch("sys.argv", test_args),
                patch("backend.etl.orchestrator.logger"),
                patch("sys.exit") as mock_exit,
            ):
                await main()

                # Should exit with code 0 for successful dry run
                mock_exit.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_main_function_with_mocked_orchestrator(self):
        """Test main function with mocked orchestrator."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            test_args = [
                "orchestrator",
                "--config",
                str(config_path),
                "--skip-scraping",
                "--download-limit",
                "10",
            ]

            with (
                patch("sys.argv", test_args),
                patch("backend.etl.orchestrator.logger"),
                patch(
                    "backend.etl.orchestrator.ETLOrchestrator"
                ) as mock_orchestrator_class,
                patch("sys.exit") as mock_exit,
            ):
                # Mock orchestrator instance
                mock_orchestrator = AsyncMock()
                mock_orchestrator.run_pipeline.return_value = [
                    ComponentResult(
                        component_name="test",
                        status=ComponentStatus.COMPLETED,
                        start_time=100.0,
                        end_time=110.0,
                        error=None,
                        metrics={"processed": 1},
                        output_files=[],
                    )
                ]
                mock_orchestrator_class.return_value = mock_orchestrator

                await main()

                # Verify orchestrator was created and pipeline was run
                mock_orchestrator_class.assert_called_once()
                mock_orchestrator.run_pipeline.assert_called_once()
                mock_exit.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_main_function_with_pipeline_failure(self):
        """Test main function handling pipeline failures."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("test_config: true")

            test_args = [
                "orchestrator",
                "--config",
                str(config_path),
            ]

            with (
                patch("sys.argv", test_args),
                patch("backend.etl.orchestrator.logger"),
                patch(
                    "backend.etl.orchestrator.ETLOrchestrator"
                ) as mock_orchestrator_class,
                patch("sys.exit") as mock_exit,
            ):
                # Mock orchestrator with failed results
                mock_orchestrator = AsyncMock()
                mock_orchestrator.run_pipeline.return_value = [
                    ComponentResult(
                        component_name="failed_component",
                        status=ComponentStatus.FAILED,
                        start_time=100.0,
                        end_time=110.0,
                        error="Component failed",
                        metrics={},
                        output_files=[],
                    )
                ]
                mock_orchestrator_class.return_value = mock_orchestrator

                await main()

                # Should exit with code 1 for pipeline failure
                mock_exit.assert_called_once_with(1)
