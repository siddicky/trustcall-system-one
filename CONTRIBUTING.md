# Contributing

Read [AGENTS.md](AGENTS.md) first: it has the rules and the module layout. The short version:
install with `make install`, and run `make lint` and `make test` before every commit. `make help`
lists every target.

## Releasing

Releases go to PyPI from a git tag, through PyPI trusted publishing, so no API token is stored in
the repository or in GitHub.

1. Set the new version in `pyproject.toml` (`version`) and in `trustcall_system_one/_version.py`
   (`__version__`). Both must match, and the tag must match both.
2. Merge the change to `main` with CI green.
3. Tag that commit and push the tag, for example:

   ```bash
   git tag v0.1.0a1
   git push origin v0.1.0a1
   ```

4. The `Release` workflow (`.github/workflows/release.yml`) builds the sdist and wheel, fails if
   the tag does not match the `pyproject.toml` version, and publishes with
   `uv publish --trusted-publishing always` in the `pypi` environment.

One-time setup, before the first release: on pypi.org, add a pending trusted publisher for
`trustcall-system-one` with owner `siddicky`, repository `trustcall-system-one`, workflow
`release.yml` and environment `pypi`. The GitHub repository needs an environment named `pypi`
(Settings > Environments).
