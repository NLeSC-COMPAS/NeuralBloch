"""Build script that also installs Julia and the packages of juliapkg.json."""

import json
import os
import sys
import sysconfig
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).parent.resolve()
JULIAPKG_JSON = ROOT / "src" / "neuralbloch" / "juliapkg.json"


class BuildPyWithJulia(build_py):
    """build_py that also resolves the Julia environment."""

    def run(self) -> None:
        """Build the modules, then install Julia into the target environment."""
        super().run()

        if os.environ.get("NEURALBLOCH_SKIP_JULIA"):
            return

        sys.path.insert(0, str(JULIAPKG_JSON.parent))
        import juliapkg

        juliapkg.resolve()
        self.point_meta_at_install()

    def point_meta_at_install(self) -> None:
        """Make juliapkg treat the build-time install as done at runtime."""
        from juliapkg.state import STATE

        meta_file = Path(STATE["meta"])
        meta = json.loads(meta_file.read_text())
        if "deps_files" not in meta:
            return
        site_packages = Path(sysconfig.get_paths()["purelib"])

        deps_files = {}
        for name, info in meta["deps_files"].items():
            path = Path(name)
            # An editable install uses juliapkg.json of the source tree
            if path == JULIAPKG_JSON and not self.editable_mode:
                path = site_packages / "neuralbloch" / "juliapkg.json"
            elif path.parent.parent.name == "site-packages":
                path = site_packages / path.parent.name / "juliapkg.json"
            # Timestamp 0 makes juliapkg compare the hash of each file
            deps_files[os.path.normcase(str(path))] = {**info, "timestamp": 0}
        meta["deps_files"] = deps_files
        meta_file.write_text(json.dumps(meta))


setup(cmdclass={"build_py": BuildPyWithJulia})
