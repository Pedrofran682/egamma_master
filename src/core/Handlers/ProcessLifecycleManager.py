import logging
import signal
import sys
import types
from typing import Any, Callable, Dict, Optional


class ProcessLifecycleManager:
    """Manages process lifecycle events, external POSIX signals, and uncaught exceptions.

    Attributes:
        logger: Logger instance utilized to record lifecycle events.
        installed: Boolean flag tracking active registration of hooks.
        original_signal_handlers: Stored previous signal handlers for clean restoration.
        original_excepthook: Stored previous sys.excepthook for clean restoration.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initializes ProcessLifecycleManager.

        Args:
            logger: Optional logger instance. Uses root logger if omitted.
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.installed: bool = False
        self.original_signal_handlers: Dict[signal.Signals, Any] = {}
        self.original_excepthook: Optional[Callable[..., Any]] = None

    def install(self) -> None:
        """Registers handlers for termination signals and uncaught exceptions."""
        if self.installed:
            return

        target_signals = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]
        for sig in target_signals:
            try:
                self.original_signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self.handle_signal)
            except (ValueError, OSError) as error:
                self.logger.warning(f"Could not attach handler for signal {sig.name}: {error}")

        self.original_excepthook = sys.excepthook
        sys.excepthook = self.handle_exception
        self.installed = True
        self.logger.debug("ProcessLifecycleManager successfully installed.")

    def uninstall(self) -> None:
        """Restores original signal handlers and system excepthook."""
        if not self.installed:
            return

        for sig, original_handler in self.original_signal_handlers.items():
            try:
                signal.signal(sig, original_handler)
            except (ValueError, OSError):
                pass
        self.original_signal_handlers.clear()

        if self.original_excepthook is not None:
            sys.excepthook = self.original_excepthook
            self.original_excepthook = None

        self.installed = False
        self.logger.debug("ProcessLifecycleManager uninstalled.")

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handles captured termination signals, logs diagnosis, and terminates process.

        Args:
            signum: Signal number received.
            frame: Execution frame at the moment of signal interception.
        """
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"UNKNOWN_{signum}"

        line_number = frame.f_lineno if frame else "unknown"
        filename = frame.f_code.co_filename if frame else "unknown"

        self.logger.critical(
            f"Process received termination signal: {signal_name} ({signum}) at {filename}:{line_number}. "
            "Flushing logs and shutting down."
        )
        logging.shutdown()
        sys.exit(128 + signum)

    def handle_exception(
        self,
        exc_type: type,
        exc_value: BaseException,
        exc_traceback: Optional[types.TracebackType],
    ) -> None:
        """Intercepts uncaught exceptions and logs full trace before bubbling.

        Args:
            exc_type: Type of the unhandled exception.
            exc_value: Unhandled exception instance.
            exc_traceback: Traceback object.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            if self.original_excepthook:
                self.original_excepthook(exc_type, exc_value, exc_traceback)
            return

        self.logger.critical(
            "Uncaught exception encountered in process lifecycle:",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        logging.shutdown()
        if self.original_excepthook:
            self.original_excepthook(exc_type, exc_value, exc_traceback)
