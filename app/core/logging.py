import json
import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

var_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def set_request_id(request_id: Optional[str]) -> None:
    var_request_id.set(request_id)


def get_request_id() -> Optional[str]:
    return var_request_id.get()


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = var_request_id.get()
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "file": f"{record.pathname}:{record.lineno}",
        }

        if getattr(record, "request_id", None):
            log_data["request_id"] = record.request_id

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or "-"
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} [{record.levelname}] [{request_id}] {record.name}: {record.getMessage()}"
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
        return log_message


def setup_logging(debug_mode: bool = False, log_level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.addFilter(ContextFilter())

    if debug_mode:
        handler.setFormatter(ConsoleFormatter())
    else:
        handler.setFormatter(JSONFormatter())

    root_logger.addHandler(handler)


logger = logging.getLogger("embedding_service")
