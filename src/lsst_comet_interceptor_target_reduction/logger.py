import logging

from colorama import Fore, Style, init

# Define colors for each log level
COLORS = {
    # 'INFO': Fore.CYAN + Style.BRIGHT,
    "DEBUG": Fore.GREEN + Style.BRIGHT,
    "WARNING": Fore.YELLOW + Style.BRIGHT,
    "ERROR": Fore.RED + Style.BRIGHT,
    "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
}


class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages based on their level.
    """

    def format(self, record):
        """
        Format the log record with colors based on the log level.
        """
        # Get the color for the specific log level
        log_color = COLORS.get(record.levelname, "")
        log_message = super().format(record)
        return f"{log_color}{log_message}{Style.RESET_ALL}"


class Logger:
    """
    A logger class that outputs colored log messages to the console.
    """

    def __init__(self, level=logging.INFO):
        # Initialize Colorama for cross-platform colors
        init(autoreset=True)

        # Configure the main logger
        self.logger = logging.getLogger("ColorLogger")
        self.logger.setLevel(level)

        # Configure the stream handler (console)
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))

        # Add the handler only once
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def set_level(self, level=logging.INFO):
        """Change the logging level."""
        self.logger.setLevel(level)

    def info(self, message):
        """Log an informational message."""
        self.logger.info(message)

    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)

    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message):
        """Log a critical message."""
        self.logger.critical(message)
