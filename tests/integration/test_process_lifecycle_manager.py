import logging
import signal
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.core.Handlers.ProcessLifecycleManager import ProcessLifecycleManager


class TestProcessLifecycleManager(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.manager = ProcessLifecycleManager(logger=self.mock_logger)

    def tearDown(self):
        self.manager.uninstall()

    def test_install_and_uninstall(self):
        initial_excepthook = sys.excepthook
        self.manager.install()
        self.assertTrue(self.manager.installed)
        self.assertEqual(sys.excepthook, self.manager.handle_exception)

        self.manager.uninstall()
        self.assertFalse(self.manager.installed)
        self.assertEqual(sys.excepthook, initial_excepthook)

    def test_handle_signal_logs_and_exits(self):
        frame = MagicMock()
        frame.f_lineno = 42
        frame.f_code.co_filename = "test_module.py"

        with self.assertRaises(SystemExit) as exit_context:
            self.manager.handle_signal(signal.SIGTERM, frame)

        self.assertEqual(exit_context.exception.code, 128 + signal.SIGTERM)
        self.mock_logger.critical.assert_called_once()
        log_message = self.mock_logger.critical.call_args[0][0]
        self.assertIn("SIGTERM", log_message)
        self.assertIn("test_module.py:42", log_message)

    def test_handle_exception_logs_critical(self):
        original_hook = MagicMock()
        self.manager.original_excepthook = original_hook

        try:
            raise ValueError("Test unexpected error")
        except ValueError as err:
            exc_info = sys.exc_info()
            self.manager.handle_exception(exc_info[0], exc_info[1], exc_info[2])

        self.mock_logger.critical.assert_called_once()
        self.assertIn("Uncaught exception", self.mock_logger.critical.call_args[0][0])
        original_hook.assert_called_once()


    def test_subprocess_signal_interception(self):
        import subprocess
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_lifecycle.log")
            script = f"""
import logging, time
from src.core.Handlers.ProcessLifecycleManager import ProcessLifecycleManager

logger = logging.getLogger("TestSubprocess")
handler = logging.FileHandler(r"{log_path}")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

manager = ProcessLifecycleManager(logger=logger)
manager.install()
print("READY", flush=True)
while True:
    time.sleep(0.1)
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            line = proc.stdout.readline()
            self.assertIn("READY", line)
            proc.terminate()
            exit_code = proc.wait(timeout=5)
            self.assertEqual(exit_code, 128 + signal.SIGTERM)

            with open(log_path, "r") as f:
                content = f.read()
            self.assertIn("Process received termination signal: SIGTERM (15)", content)


if __name__ == "__main__":
    unittest.main()
