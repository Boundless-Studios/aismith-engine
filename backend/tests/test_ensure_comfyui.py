import unittest

import requests

from backend.app.image_preparation import ensure_comfyui


class EnsureComfyuiTests(unittest.TestCase):
    def test_reachable_server_is_left_alone(self):
        started = []
        ensure_comfyui("http://x", start=lambda: started.append(1), reachable=lambda _s: True, sleep=lambda _t: None)
        self.assertEqual(started, [])

    def test_stopped_server_is_started_and_waited_for(self):
        started = []
        answers = iter([False, False, True])
        slept = []
        ensure_comfyui(
            "http://x", start=lambda: started.append(1), reachable=lambda _s: next(answers),
            sleep=slept.append, timeout=60.0,
        )
        self.assertEqual(started, [1])
        self.assertEqual(slept, [3.0])  # one poll gap: down at the first check after starting, up at the next

    def test_a_missing_systemctl_surfaces_as_comfyui_unavailable(self):
        # macOS / CI / a container: no `systemctl` binary. That is the plain
        # "ComfyUI is down" condition, and callers translate ConnectionError
        # into their "ComfyUI unavailable" message -- a FileNotFoundError
        # would bypass that handler.
        def start():
            raise FileNotFoundError(2, "No such file or directory", "systemctl")

        with self.assertRaises(requests.exceptions.ConnectionError) as ctx:
            ensure_comfyui("http://x", start=start, reachable=lambda _s: False, sleep=lambda _t: None, timeout=0.0)
        self.assertIn("http://x", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, FileNotFoundError)

    def test_server_that_never_answers_raises_within_the_timeout(self):
        with self.assertRaises(RuntimeError) as ctx:
            ensure_comfyui("http://x", start=lambda: None, reachable=lambda _s: False, sleep=lambda _t: None, timeout=0.0)
        self.assertIn("did not come up", str(ctx.exception))
