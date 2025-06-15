"""
Logging configuration for Credit Default Prediction
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "credit_default",
    log_file: str = None,
    level: int = logging.INFO,
    format_string: str = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        format_string: Log format string

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "credit_default") -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Create default logger
def create_default_logger():
    """Create default logger for the project"""
    log_dir = Path(__file__).parent.parent.parent.parent / "logs"
    log_file = log_dir / f"credit_default_{datetime.now().strftime('%Y%m%d')}.log"

    return setup_logger(
        name="credit_default",
        log_file=str(log_file),
        level=logging.INFO
    )


# Initialize default logger
logger = create_default_logger()
