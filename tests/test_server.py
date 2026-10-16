import functools
from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen, Request

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("studio_server", ROOT / "serve.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QuietHandler(module.StudioHandler):
    def log_message(self, *_args):
        pass


class StaticServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(ROOT)))
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_public_app_files_and_guide(self):
        for path in ("/", "/index.html", "/gvhmr.html", "/GVHMR%20Setup.md", "/rtmw3d.html", "/RTMW3D.md", "/app.mjs?cache=1", "/blender_importer.py"):
            with urlopen(self.base + path) as response:
                self.assertEqual(response.status, 200)

    def test_models_environments_caches_and_traversal_are_not_served(self):
        for path in ("/.third-party/GVHMR/LICENSE", "/.venv-gvhmr/pyvenv.cfg", "/.gvhmr-runs/",
                     "/.models/rtmw3d/rtmw3d-x.onnx", "/.venv-rtmw3d/pyvenv.cfg",
                     "/backends/gvhmr/runner.py", "/node_modules/", "/%2e%2e/AGENTS.md", "/tests/fixtures/test.webm"):
            for method in ("GET", "HEAD"):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(self.base + path, method=method))
                self.assertEqual(error.exception.code, 404)
                error.exception.close()


if __name__ == "__main__":
    unittest.main()
