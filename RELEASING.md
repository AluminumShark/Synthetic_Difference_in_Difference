# Releasing

Releases are fully automated by GitHub Actions and use **no API tokens** — uploads
authenticate with PyPI trusted publishing (OIDC). The pipeline lives in
[`.github/workflows/release.yml`](.github/workflows/release.yml).

```
git push origin v0.4.1
        |
        |- build        version vs tag check -> pytest -> uv build -> twine check
        |- TestPyPI     upload to the test index
        |- smoke-test   pip install from TestPyPI and import the package
        |- PyPI         upload to the production index (can require manual approval)
        '- Release      create the GitHub Release with the same dist files
```

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

### 2. TestPyPI pending publisher
The name `sdid` is not registered on TestPyPI, so there is no project settings page yet.
Use a *pending* publisher instead: <https://test.pypi.org/manage/account/publishing/>,
set PyPI Project Name to `sdid`, the rest as above but with Environment name `testpypi`.
It converts to a normal publisher after the first successful upload.

### 3. GitHub Environments
Repo -> Settings -> Environments, create two environments (names must match the above exactly):

- `testpypi`
- `pypi` — enable **Required reviewers** (add yourself). The workflow then pauses for
  approval before touching the production index, which is the last safety net against
  an accidental tag push.

---

## Per release

1. Bump the version in **one place only**: `__version__` in `src/sdid/__init__.py`.
   `pyproject.toml` declares `dynamic = ["version"]` with `[tool.hatch.version]` and reads
   it from there, so there is nothing to keep in sync.
2. `uv lock` -> `uv run pytest` -> `uv run ruff check .`
3. Commit, push to `main`, and wait for CI to pass.
4. Tag and push:
   ```bash
   git tag -a v0.4.1 -m "v0.4.1"
   git push origin v0.4.1
   ```
5. Watch the Release workflow in the Actions tab. If the `pypi` environment requires a
   reviewer, approve it there (**Review deployments -> Approve**).

> A version number can only be used once: neither PyPI nor TestPyPI allows overwriting an
> existing version. If a release goes wrong, move to the next version (0.4.1 -> 0.4.2).

## Rehearsing without touching production

Actions -> **Release** -> *Run workflow* -> set `target` to `testpypi`.
This runs build, the TestPyPI upload and the smoke test, then stops: no PyPI upload and
no GitHub Release.

## Troubleshooting

| Message | Cause |
| --- | --- |
| `invalid-publisher` / OIDC 403 | One of owner / repo / workflow filename / environment does not match the publisher registered on PyPI |
| `Missing id-token permission` | The job is missing `permissions: id-token: write` |
| `File already exists` | That version was already uploaded; bump the version |
| `tag v0.4.1 does not match package version` | The tag disagrees with `__version__` in `src/sdid/__init__.py` |
| `Unable to resolve action ...` | A pinned action version does not exist (note that `astral-sh/setup-uv` has no floating major tag) |
