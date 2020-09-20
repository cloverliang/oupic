import shlex
from pathlib import Path
from platform import python_version

import invoke

PACKAGE = "pylce"
REQUIRED_COVERAGE = 0


@invoke.task(help={"python": "Force a python version (default: current version)"})
def bootstrap(ctx, python=python_version()):
    """Install required conda packages."""

    def ensure_packages(*packages):
        clean_packages = sorted({shlex.quote(package) for package in sorted(packages)})
        ctx.run(
            "conda install --quiet --yes " + " ".join(clean_packages),
            pty=True,
            echo=True,
        )

    try:
        import jinja2
        import yaml
    except ModuleNotFoundError:
        ensure_packages("jinja2", "pyyaml")
        import jinja2
        import yaml

    with open("meta.yaml") as file:
        template = jinja2.Template(file.read())

    meta_yaml = yaml.safe_load(
        template.render(load_setup_py_data=lambda: {}, python=python)
    )
    develop_packages = meta_yaml["requirements"]["develop"]
    build_packages = meta_yaml["requirements"]["build"]
    run_packages = meta_yaml["requirements"]["run"]

    ensure_packages(*develop_packages, *build_packages, *run_packages)


@invoke.task(
    help={"all": f"Remove {PACKAGE}.egg-info directory too", "n": "Dry-run mode"}
)
def clean(ctx, all_=False, n=False):
    """Clean unused files."""
    args = ["-d", "-x", "-e .idea"]
    if not all_:
        args.append(f"-e {PACKAGE}.egg-info")
    args.append("-n" if n else "-f")
    ctx.run("git clean " + " ".join(args), echo=True)


@invoke.task(
    incrementable=["verbose"],
    help={
        "behavioral": "Run behavioral tests too (default: False)",
        "performance": "Run performance tests too (default: False)",
        "external": "Run external tests too (default: False)",
        "x": "Exit instantly on first error or failed test (default: False)",
        "junit-xml": "Create junit-xml style report (default: False)",
        "failed-first": "run all tests but run the last failures first (default: False)",
        "quiet": "Decrease verbosity",
        "verbose": "Increase verbosity (can be repeated)",
    },
)
def test(
    ctx,
    behavioral=False,
    performance=False,
    external=False,
    x=False,
    junit_xml=False,
    failed_first=False,
    quiet=False,
    verbose=0,
):
    """Run tests."""
    markers = []
    if not behavioral:
        markers.append("not behavioral")
    if not performance:
        markers.append("not performance")
    if not external:
        markers.append("not external")
    args = []
    if markers:
        args.append("-m '" + " and ".join(markers) + "'")
    if not behavioral and not performance and not external:
        args.append(f"--cov={PACKAGE}")
        args.append(f"--cov-fail-under={REQUIRED_COVERAGE}")
    if x:
        args.append("-x")
    if junit_xml:
        args.append("--junit-xml=junit.xml")
    if failed_first:
        args.append("--failed-first")
    if quiet:
        verbose -= 1
    if verbose < 0:
        args.append("--quiet")
    if verbose > 0:
        args.append("-" + ("v" * verbose))
    ctx.run("pytest " + " ".join(args), pty=True, echo=True)


@invoke.task(
    help={
        "style": "Check style with flake8, isort, and black",
        "typing": "Check typing with mypy",
    }
)
def check(ctx, style=True, typing=True):
    """Check for style and static typing errors."""
    if style:
        ctx.run(f"flake8 {PACKAGE} tests", echo=True)
        ctx.run(f"isort --diff {PACKAGE} tests --check-only", echo=True)
        ctx.run(f"black --diff {PACKAGE} tests --check", echo=True)
    if typing:
        ctx.run(f"mypy --no-incremental --cache-dir=/dev/null {PACKAGE}", echo=True)


@invoke.task(name="format", aliases=["fmt"])
def format_(ctx):
    """Format code to use standard style guidelines."""
    autoflake = "autoflake -i --recursive --remove-all-unused-imports --remove-duplicate-keys --remove-unused-variables"
    ctx.run(f"{autoflake} {PACKAGE} tests", echo=True)
    ctx.run(f"isort {PACKAGE} tests", echo=True)
    ctx.run(f"black {PACKAGE} tests", echo=True)


@invoke.task
def install(ctx):
    """Install the package."""
    ctx.run("python -m pip install .", echo=True)


@invoke.task
def develop(ctx):
    """Install the package in editable mode."""
    ctx.run("python -m pip install --no-use-pep517 --editable .", echo=True)


@invoke.task(aliases=["undevelop"])
def uninstall(ctx):
    """Uninstall the package."""
    ctx.run(f"python -m pip uninstall --yes {PACKAGE}", echo=True)


@invoke.task
def hooks(ctx, uninstall_=False):
    """Install (or uninstall) git hooks."""

    def install_hooks():
        invoke_path = Path(ctx.run("which invoke", hide=True).stdout[:-1])
        for src_path in Path(".hooks").iterdir():
            dst_path = Path(".git/hooks") / src_path.name
            print(f"Installing: {dst_path}")
            with open(str(src_path), "r") as f:
                src_data = f.read()
            with open(str(dst_path), "w") as f:
                f.write(src_data.format(invoke_path=invoke_path.parent))
            ctx.run(f"chmod +x {dst_path}")

    def uninstall_hooks():
        for path in Path(".git/hooks").iterdir():
            if not path.suffix:
                print(f"Uninstalling: {path}")
                path.unlink()

    if uninstall_:
        uninstall_hooks()
    else:
        install_hooks()


@invoke.task
def docs(ctx):
    """Generate package documentation."""
    version = ctx.run("python setup.py --version", hide=True).stdout.split("\n")[-2]
    # ctx.run("rm -rf dist/docs")
    # ctx.run(f"sphinx-build -W --keep-going -D 'version={version}' docs dist/docs/{version}",
    #         echo=True)
    print(Path(f"dist/docs/{version}/index.html").absolute())


@invoke.task(
    help={
        "python": "Force a python version (default: current version)",
        "convert": "Convert package to windows too",
    }
)
def build(ctx, python=python_version(), convert=True):
    """Build conda package(s)."""
    ctx.run("rm -rf conda.dist")
    ctx.run("mkdir -p conda.dist")
    conda_build = f"conda build --quiet --no-include-recipe --output-folder=conda.dist --no-test --python {python} ."
    ctx.run(conda_build, pty=True, echo=True)
    ctx.run("chmod 777 conda.dist/linux-64/*.tar.bz2", echo=True)
    ctx.run("rm -rf dist/linux-64")
    ctx.run("mkdir -p dist/linux-64")
    ctx.run("cp conda.dist/linux-64/*.tar.bz2 dist/linux-64", echo=True)
    if convert:
        ctx.run("rm -rf dist/win-64")
        ctx.run("mkdir -p dist/win-64")
        ctx.run(
            "conda convert --force --platform win-64 --output-dir dist dist/linux-64/*.tar.bz2",
            echo=True,
        )
        ctx.run("chmod 777 dist/win-64/*.tar.bz2", echo=True)


@invoke.task(
    help={"linux": "Verify Linux package", "windows": "Verify Windows package"}
)
def verify(ctx, linux=True, windows=True):
    """Verify conda package(s)."""
    version = ctx.run("python setup.py --version", hide=True).stdout.split("\n")[-2]
    if linux:
        ctx.run(
            f"tar -jtvf dist/linux-64/{PACKAGE}-{version}-*.tar.bz2 | sort -k 6",
            echo=True,
        )
        ctx.run(f"conda verify dist/linux-64/{PACKAGE}-{version}-*.tar.bz2", echo=True)
    if windows:
        ctx.run(
            f"tar -jtvf dist/win-64/{PACKAGE}-{version}-*.tar.bz2 | sort -k 6",
            echo=True,
        )
        ctx.run(f"conda verify dist/win-64/{PACKAGE}-{version}-*.tar.bz2", echo=True)
