import logging
import os


def setup_singleton_logger():
    """Configures the 'Restorationist' logger as a singleton."""
    logger = logging.getLogger("chatAsYou260325")

    # If handlers already exist, don't add them again
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Formatter for structured parsing
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

        # Console Handler for real-time monitoring
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler for future upgrades/parsing
        fh = logging.FileHandler("agent_workflow.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
