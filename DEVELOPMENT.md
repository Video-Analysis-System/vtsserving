# Developer Guide

Before getting started, check out the `#VTSSERVING-contributors` channel in the [VTSSERVING community slack](https://l.linklyhq.com/l/ktOh).

If you are interested in contributing to existing issues and feature requets, check out the [good-first-issue](https://github.com/VTSSERVING/VTSSERVING/issues?q=is%3Aopen+is%3Aissue+label%3Agood-first-issue) and [help-wanted](https://github.com/VTSSERVING/VTSSERVING/issues?q=is%3Aopen+is%3Aissue+label%3Ahelp-wanted) issues list.

If you are interested in proposing a new feature, make sure to create a new feature request ticket [here](https://github.com/VTSSERVING/VTSSERVING/issues/new/choose) and share your proposal in the `#VTSSERVING-contributors` slack channel for feedback.

## Start Developing

<details><summary><h3>with the Command Line</h3></summary>

1. Make sure to have [Git](https://git-scm.com/), [pip](https://pip.pypa.io/en/stable/installation/), and [Python3.7+](https://www.python.org/downloads/) installed.

   Optionally, make sure to have [GNU Make](https://www.gnu.org/software/make/) available on your system if you aren't using a UNIX-based system for a better developer experience.
   If you don't want to use `make` then please refer to the [Makefile](./Makefile) for specific commands on a given make target.

2. Fork the VTSSERVING project on [GitHub](https://github.com/VTSSERVING/VTSSERVING).

3. Clone the source code from your fork of VTSSERVING's GitHub repository:

   ```bash
   git clone git@github.com:username/vtsserving.git && cd VTSSERVING
   ```

4. Add the VTSSERVING upstream remote to your local VTSSERVING clone:

   ```bash
   git remote add upstream git@github.com:VTSSERVING/vtsserving.git
   ```

5. Configure git to pull from the upstream remote:

   ```bash
   git switch main # ensure you're on the main branch
   git branch --set-upstream-to=upstream/main
   ```

6. Install VTSSERVING with pip in editable mode:

   ```bash
   pip install -e .
   ```

   This installs VTSSERVING in an editable state. The changes you make will automatically be reflected without reinstalling vtsserving.

7. Install the VTSSERVING development requirements:

   ```bash
   pip install -r ./requirements/dev-requirements.txt
   ```

8. Test the VTSSERVING installation either with `bash`:

   ```bash
   VTSSERVING --version
   ```

   or in a Python session:

   ```python
   import VTSSERVING
   print(vtsserving.__version__)
   ```

</details>

<details><summary><h3>with VS Code</h3></summary>

1. Confirm that you have the following installed:

   - [Python3.7+](https://www.python.org/downloads/)
   - VS Code with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) extensions

2. Fork the VTSSERVING project on [GitHub](https://github.com/VTSSERVING/VTSSERVING).

3. Clone the GitHub repository:

   1. Open the command palette with Ctrl+Shift+P and type in 'clone'.
   2. Select 'Git: Clone(Recursive)'.
   3. Clone vtsserving.

4. Add an VTSSERVING upstream remote:

   1. Open the command palette and enter 'add remote'.
   2. Select 'Git: Add Remote'.
   3. Press enter to select 'Add remote' from GitHub.
   4. Enter https://github.com/VTSSERVING/vtsserving.git to select the VTSSERVING repository.
   5. Name your remote 'upstream'.

5. Pull from the VTSSERVING upstream remote to your main branch:

   1. Open the command palette and enter 'checkout'.
   2. Select 'Git: Checkout to...'
   3. Choose 'main' to switch to the main branch.
   4. Open the command palette again and enter 'pull from'.
   5. Click on 'Git: Pull from...'
   6. Select 'upstream'.

6. Open a new terminal by clicking the Terminal dropdown at the top of the window, followed by the 'New Terminal' option. Next, add a virtual environment with this command:
   ```bash
   python -m venv .venv
   ```
7. Click yes if a popup suggests switching to the virtual environment. Otherwise, go through these steps:

   1. Open any python file in the directory.
   2. Select the interpreter selector on the blue status bar at the bottom of the editor.
      ![vscode-status-bar](https://user-images.githubusercontent.com/489344/166984038-75f1f4bd-c896-43ee-a7ee-1b57fda359a3.png)

   3. Switch to the path that includes .venv from the dropdown at the top.
      ![vscode-select-venv](https://user-images.githubusercontent.com/489344/166984060-170d25f5-a91f-41d3-96f4-4db3c21df7c8.png)

8. Update your PowerShell execution policies. Win+x followed by the 'a' key opens the admin Windows PowerShell. Enter the following command to allow the virtual environment activation script to run:
   ```
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   </details>

## Making Changes

<details><summary><h3>using the Command Line</h3></summary>

1. Make sure you're on the main branch.

   ```bash
   git switch main
   ```

2. Use the git pull command to retrieve content from the VTSSERVING Github repository.

   ```bash
   git pull
   ```

3. Create a new branch and switch to it.

   ```bash
   git switch -c my-new-branch-name
   ```

4. Make your changes!

5. Use the git add command to save the state of files you have changed.

   ```bash
   git add <names of the files you have changed>
   ```

6. Commit your changes.

   ```bash
   git commit
   ```

7. Push all changes to your fork on GitHub.
   ```bash
   git push
   ```
   </details>

<details><summary><h3>using VS Code</h3></summary>

1. Switch to the main branch:

   1. Open the command palette with Ctrl+Shift+P.
   2. Search for 'Git: Checkout to...'
   3. Select 'main'.

2. Pull from the upstream remote:

   1. Open the command palette.
   2. Enter and select 'Git: Pull...'
   3. Select 'upstream'.

3. Create and change to a new branch:

   1. Type in 'Git: Create Branch...' in the command palette.
   2. Enter a branch name.

4. Make your changes!

5. Stage all your changes:

   1. Enter and select 'Git: Stage All Changes...' in the command palette.

6. Commit your changes:

   1. Open the command palette and enter 'Git: Commit'.

7. Push your changes:
   1. Enter and select 'Git: Push...' in the command palette.

</details>

## Run VTSSERVING with verbose/debug logging

To view internal debug loggings for development, set the `VTSSERVING_DEBUG` environment variable to `TRUE`:

```bash
export VTSSERVING_DEBUG=TRUE
```

And/or use the `--verbose` option when running `VTSSERVING` CLI command, e.g.:

```bash
VTSSERVING get IrisClassifier --verbose
```

## Style check, auto-formatting, type-checking

formatter: [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort), [buf](https://github.com/bufbuild/buf)

linter: [pylint](https://pylint.org/), [buf](https://github.com/bufbuild/buf)

type checker: [pyright](https://github.com/microsoft/pyright)

We are using [buf](https://github.com/bufbuild/buf) for formatting and linting
of our proto files. Configuration can be found [here](./VTSSERVING/grpc/buf.yaml).
Currently, we are running `buf` with docker, hence we kindly ask our developers
to have docker available. Docker installation can be found [here](https://docs.docker.com/get-docker/).

Run linter/format script:

```bash
make format

make lint
```

Run type checker:

```bash
make type
```

## Editing proto files

The proto files for the VTSSERVING gRPC service are located under [`VTSSERVING/grpc`](./VTSSERVING/grpc/).
The generated python files are not checked in the git repository, and are instead generated via this [`script`](./scripts/generate_grpc_stubs.sh).
If you edit the proto files, make sure to run `./scripts/generate_grpc_stubs.sh` to
regenerate the proto stubs.

## Deploy with your changes

Test test out your changes in an actual VTSSERVING model deployment, you can create a new Vts with your custom VTSSERVING source repo:

1. Install custom VTSSERVING in editable mode. e.g.:
   - git clone your VTSSERVING fork
   - `pip install -e PATH_TO_THE_FORK`
2. Set env var `export VTSSERVING_BUNDLE_LOCAL_BUILD=True` and `export SETUPTOOLS_USE_DISTUTILS=stdlib`
   - make sure you have the latest setuptools installed: `pip install -U setuptools`
3. Build a new Vts with `VTSSERVING build` in your project directory
4. The new Vts will include a wheel file built from the VTSSERVING source, and
   `VTSSERVING containerize` will install it to override the default VTSSERVING installation in base image

### Distribute a custom VTSSERVING release for your team

If you want other team members to easily use your custom VTSSERVING distribution, you may publish your
branch to your fork of VTSSERVING, and have your users install it this way:

```bash
pip install git+https://github.com/{YOUR_GITHUB_USERNAME}/VTSSERVING@{YOUR_REVISION}
```

And in your VTSSERVING projects' `vtsfile.yaml`, force the Vts to install this distribution, e.g.:

```yaml
service: "service:svc"
description: "file: ./README.md"
include:
  - "*.py"
python:
  packages:
    - pandas
    - git+https://github.com/{YOUR_GITHUB_USERNAME}/VTSSERVING@{YOUR_REVISION}
docker:
  system_packages:
    - git
```

## Testing

Make sure to install all test dependencies:

```bash
pip install -r requirements/tests-requirements.txt
```

VTSSERVING tests come with a Pytest plugin. Export `PYTEST_PLUGINS`:

```bash
export PYTEST_PLUGINS=vtsserving.testing.pytest.plugin
```

### Unit tests

You can run unit tests in two ways:

Run all unit tests directly with pytest:

```bash
# GIT_ROOT=$(git rev-parse --show-toplevel)
pytest tests/unit
```

### Integration tests

Write a general framework tests under `./tests/integration/frameworks/models`, and the
run the following command

```bash
pytest tests/integration/frameworks/test_frameworks.py --framework pytorch
```

### E2E tests

```bash
# example: run e2e tests to check for http general features
pytest tests/e2e/vts_server_grpc
```

### Adding new test suite

If you are adding new ML framework support, it is recommended that you also add a separate test suite in our CI. Currently we are using GitHub Actions to manage our CI/CD workflow.

We recommend using [`nektos/act`](https://github.com/nektos/act) to run and test Actions locally.

Add a new job for your new framework under [framework.yml](./.github/workflows/frameworks.yml)

## Python tools ecosystem

Currently, VTSSERVING is [PEP518](https://www.python.org/dev/peps/pep-0518/) compatible. We define package configuration via [`pyproject.toml`][https://github.com/VTSSERVING/VTSSERVING/blob/main/pyproject.toml].

## Benchmark

VTSSERVING has moved its benchmark to [`VTSSERVING/benchmark`](https://github.com/VTSSERVING/benchmark).

## Creating Pull Requests on GitHub

Push changes to your fork and follow [this
article](https://help.github.com/en/articles/creating-a-pull-request)
on how to create a pull request on github. Name your pull request
with one of the following prefixes, e.g. "feat: add support for
PyTorch". This is based on the [Conventional Commits
specification](https://www.conventionalcommits.org/en/v1.0.0/#summary)

- feat: (new feature for the user, not a new feature for build script)
- fix: (bug fix for the user, not a fix to a build script)
- docs: (changes to the documentation)
- style: (formatting, missing semicolons, etc; no production code change)
- refactor: (refactoring production code, eg. renaming a variable)
- perf: (code changes that improve performance)
- test: (adding missing tests, refactoring tests; no production code change)
- chore: (updating grunt tasks etc; no production code change)
- build: (changes that affect the build system or external dependencies)
- ci: (changes to configuration files and scripts)
- revert: (reverts a previous commit)

Once your pull request is created, an automated test run will be triggered on
your branch and the VTSSERVING authors will be notified to review your code
changes. Once tests are passed and a reviewer has signed off, we will merge
your pull request.

## Documentations

Refer to [VTSSERVING Documentation Guide](./docs/README.md) for how to build and write
docs.
