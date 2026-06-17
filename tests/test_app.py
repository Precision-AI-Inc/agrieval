# ======================================================================
#  CONFIDENTIAL — © Precision AI 2025. All Rights Reserved.
#
#  This source code and any accompanying documentation contain
#  confidential and proprietary information of Precision AI.
#
#  Unauthorized reproduction, disclosure, modification, or distribution
#  of this material is strictly prohibited and will be prosecuted to the
#  fullest extent of the law.
# ======================================================================

"""Unit tests for the FastAPI application factory and CLI entry point."""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from pai.ag_emb.api.app import app, main


class TestApp:
    def test_app_is_fastapi_instance(self) -> None:
        assert isinstance(app, FastAPI)

    def test_app_title(self) -> None:
        assert app.title == "pai-ag-emb"

    def test_evaluate_router_mounted(self) -> None:
        paths = set(app.openapi().get("paths", {}).keys())
        assert any("/v1/embeddings/evaluate" in p for p in paths)


class TestMain:
    @pytest.fixture(autouse=True)
    def _clean_env(self) -> Generator[None, None, None]:
        """Remove PAI_DATASET_ROOT from the environment before and after each test."""
        os.environ.pop("PAI_DATASET_ROOT", None)
        yield
        os.environ.pop("PAI_DATASET_ROOT", None)

    def _run_main(self, argv: list[str]) -> MagicMock:
        mock = MagicMock()
        with patch("uvicorn.run", mock), patch("sys.argv", ["pai-ag-emb", *argv]):
            main()
        return mock

    def test_defaults(self) -> None:
        mock = self._run_main([])
        mock.assert_called_once_with(
            "pai.ag_emb.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )

    def test_custom_host_and_port(self) -> None:
        mock = self._run_main(["--host", "127.0.0.1", "--port", "9000"])
        _, kwargs = mock.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9000

    def test_no_reload_flag(self) -> None:
        mock = self._run_main(["--no-reload"])
        _, kwargs = mock.call_args
        assert kwargs["reload"] is False

    def test_dataset_root_sets_env_var(self) -> None:
        self._run_main(["--dataset-root", "/data/crops"])
        assert os.environ.get("PAI_DATASET_ROOT") == "/data/crops"

    def test_dataset_root_default_is_dataset(self) -> None:
        self._run_main([])
        assert os.environ.get("PAI_DATASET_ROOT") == "dataset"

    def test_existing_env_var_not_overwritten(self) -> None:
        os.environ["PAI_DATASET_ROOT"] = "/existing/path"
        self._run_main([])
        assert os.environ["PAI_DATASET_ROOT"] == "/existing/path"

    def test_app_string_passed_to_uvicorn(self) -> None:
        mock = self._run_main([])
        args, _ = mock.call_args
        assert args[0] == "pai.ag_emb.api.app:app"
