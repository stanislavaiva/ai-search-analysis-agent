from __future__ import annotations

"""
Точка входа для запуска CLI-версии агента на LangGraph.
Основная конфигурация и описание запуска находятся в README и .env.example.
"""

import logging

from agent.cli import run_cli


def configure_logging() -> None:
    """Настраивает базовое логирование приложения."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    configure_logging()
    run_cli()