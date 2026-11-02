"""Build script that also installs Julia and the packages of juliapkg.json."""

import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).parent.resolve()


class BuildPyWithJulia(build_py):
    """build_py that ships juliapkg.json and resolves the Julia environment."""

    def run(self) -> None:
        """Build the modules, then install Julia into the target environment."""
        super().run()

        # FIXME: Find a better way of pointing the paths
        #
        # juliapkg looks for juliapkg.json in each sys.path entry and its
        # subdirectories. In site-packages, src/ is such a subdirectory.
        target = Path(self.build_lib) / "src"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "juliapkg.json", target / "juliapkg.json")

        if os.environ.get("NEURALBLOCH_SKIP_JULIA"):
            return

        # FIXME: Unhack this
        #
        # pip builds with the interpreter of the target environment, so
        # sys.prefix is that environment and Julia goes into its julia_env.
        sys.path.insert(0, str(ROOT))
        import juliapkg

        juliapkg.resolve()
        self.point_meta_at_install()

    @staticmethod
    def point_meta_at_install() -> None:
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
            if path == ROOT / "juliapkg.json":
                path = site_packages / "src" / "juliapkg.json"
            elif path.parent.parent.name == "site-packages":
                path = site_packages / path.parent.name / "juliapkg.json"
            # Timestamp 0 makes juliapkg compare the hash of each file
            deps_files[os.path.normcase(str(path))] = {**info, "timestamp": 0}
        meta["deps_files"] = deps_files
        meta_file.write_text(json.dumps(meta))


setup(cmdclass={"build_py": BuildPyWithJulia})
