# Releasing

Releases are automated by GitHub Actions and use **no API tokens** — uploads authenticate
with PyPI trusted publishing (OIDC). The pipeline lives in
[`.github/workflows/release.yml`](.github/workflows/release.yml).

**Merging a version bump is the release trigger.** Every push to `main` starts the
pipeline, but a gate job stops it immediately unless `__version__` has no tag yet, so an
ordinary merge costs about ten seconds and publishes nothing.

```
merge a version bump into main
        |
        |- gate         release only if v<version> is not tagged yet, otherwise stop here
        |- build        pytest -> uv build -> twine check
        |- TestPyPI     upload to the test index          [waits for your approval]
        |- smoke-test   pip install from TestPyPI and import the package
        |- PyPI         upload to the production index    [waits for your approval]
        '- Release      create the tag and the GitHub Release with the same dist files
```

Both uploads pass through an environment with you as a required reviewer, so nothing
reaches an index until you approve it in the Actions tab (**Review deployments**).

Other ways to trigger it:

- **Push a `v*` tag** — releases that tag. The gate fails the run if the tag disagrees
  with `__version__`.
- **Run it manually** (Actions -> Release -> *Run workflow*) — `target: testpypi` stops
  after the smoke test, which is the safe way to rehearse.

---

## One-time setup

### 1. PyPI trusted publisher
Go to <https://pypi.org/manage/project/sdid/settings/publishing/> -> *Add a new publisher* -> GitHub:

| Field | Value |
| --- | --- |
| Owner | `AluminumShark` |
| Repository name | `Synthetic_Difference_in_Difference` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

### 2. TestPyPI trusted publisher
Same values at <https://test.pypi.org/manage/project/sdid/settings/publishing/>, but with
Environment name `testpypi`. (Before the project existed on TestPyPI this had to be
registered as a *pending* publisher at
<https://test.pypi.org/manage/account/publishing/>; the first successful upload converted
it.)

### 3. GitHub Environments
Repo -> Settings -> Environments, with names matching the above exactly:

- `testpypi`
- `pypi`

Both have **Required reviewers** enabled. That approval is the gate that keeps the
automatic trigger safe: merging starts the pipeline, but publishing still needs a click.

---

## Per release

1. Bump the version in **one place only**: `__version__` in `src/sdid/__init__.py`.
   `pyproject.toml` declares `dynamic = ["version"]` with `[tool.hatch.version]` and reads
   it from there, so there is nothing to keep in sync.
2. `uv lock` -> `uv run pytest` -> `uv run ruff check .`
3. Open a PR, let CI pass, and merge it.
4. The Release workflow starts on the merge. Approve `testpypi`, then `pypi`, in the
   Actions tab.
5. The workflow creates the `v<version>` tag and the GitHub Release itself. Do not tag by
   hand.

> A version number can only be used once: neither PyPI nor TestPyPI allows overwriting an
> existing version. If a release goes wrong, move to the next version (0.4.1 -> 0.4.2).

## Supported Python versions

CI tests 3.10 through 3.14 on Linux, plus 3.13 and 3.14 on Windows and 3.14 on macOS.
When adding a new Python version, update three places: the `python-version` matrix in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), the
`Programming Language :: Python :: 3.x` classifiers in `pyproject.toml`, and
`requires-python` if the floor moves.

## Troubleshooting

| Message | Cause |
| --- | --- |
| Release run finished with every job skipped | The gate found `v<version>` already tagged, i.e. the merge did not change the version. Expected for ordinary merges. |
| `invalid-publisher` / OIDC 403 | One of owner / repo / workflow filename / environment does not match the publisher registered on PyPI |
| `Missing id-token permission` | The job is missing `permissions: id-token: write` |
| `File already exists` | That version was already uploaded; bump the version |
| `tag v0.4.1 does not match package version` | A pushed tag disagrees with `__version__` in `src/sdid/__init__.py` |
| `Unable to resolve action ...` | A pinned action version does not exist (note that `astral-sh/setup-uv` has no floating major tag) |
