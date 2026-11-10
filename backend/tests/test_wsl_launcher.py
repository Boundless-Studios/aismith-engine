import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]


class WslLauncherTests(unittest.TestCase):
    def test_start_loads_kimodo_and_shared_model_overrides(self):
        launcher = (ROOT / "scripts" / "start-wsl.sh").read_text(encoding="utf-8")

        for key in (
            "AISMITH_MODELS_DIR",
            "KIMODO_DIR",
            "KIMODO_EXECUTABLE",
            "KIMODO_MODEL",
            "KIMODO_TEXT_ENCODER_DEVICE",
        ):
            self.assertIn(key, launcher)

        self.assertIn("TRELLIS_REFINER_DIR", launcher)

    def test_start_binds_loopback_unless_configured_otherwise(self):
        # The API has no authentication; a LAN bind must be an explicit choice.
        launcher = (ROOT / "scripts" / "start-wsl.sh").read_text(encoding="utf-8")
        example = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

        for key in ("AISMITH_API_HOST", "AISMITH_UI_HOST"):
            self.assertIn(f"{key}=${{{key}:-127.0.0.1}}", launcher)
            self.assertIn(f"{key}=127.0.0.1", example)
        self.assertNotIn("0.0.0.0", launcher)

    def test_stop_recovers_data_directory_recorded_by_start(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "durable-state"
            marker = root / "runtime-dir"
            marker.write_text(str(data_dir), encoding="utf-8")
            environment = os.environ.copy()
            environment.pop("AISMITH_DATA_DIR", None)
            environment["AISMITH_RUNTIME_DIR_FILE"] = str(marker)

            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "stop-wsl.sh"), "--print-data-dir"],
                capture_output=True,
                check=True,
                env=environment,
                text=True,
            )

            self.assertEqual(result.stdout.strip(), str(data_dir))

    def test_stop_prefers_explicit_data_directory_over_env_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "explicit"
            env_file = root / ".env"
            env_file.write_text(f"AISMITH_DATA_DIR={root / 'configured'}\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["AISMITH_DATA_DIR"] = str(explicit)
            environment["AISMITH_ENV_FILE"] = str(env_file)

            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "stop-wsl.sh"), "--print-data-dir"],
                capture_output=True,
                check=True,
                env=environment,
                text=True,
            )

            self.assertEqual(result.stdout.strip(), str(explicit))

    def test_stop_waits_for_owned_processes_to_exit(self):
        launcher = (ROOT / "scripts" / "stop-wsl.sh").read_text(encoding="utf-8")

        self.assertIn("kill -0 \"$pid\"", launcher)


if __name__ == "__main__":
    unittest.main()
