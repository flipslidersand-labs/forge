# Release Process

Forge uses [semantic-release](https://semantic-release.gitbook.io/) to automate version bumping, changelog generation, and PyPI publishing.

## Prerequisites

### GitHub Secrets

You must configure the following GitHub secrets in your repository:

- **`PYPI_TOKEN`**: PyPI API token for automated publishing
  - Generate at: https://pypi.org/manage/account/tokens/
  - Scope: Entire account OR specific project scope

### Local Setup (Optional for manual releases)

```bash
# Install Node.js dependencies (if running locally)
npm install --save-dev semantic-release \
  @semantic-release/commit-analyzer \
  @semantic-release/release-notes-generator \
  @semantic-release/changelog \
  @semantic-release/git \
  @semantic-release/github \
  conventional-changelog-conventionalcommits
```

## Workflow

### Automatic Release (GitHub Actions)

1. **Trigger**: Merge PR to `master` branch
2. **Analysis**: semantic-release analyzes commits since last tag
3. **Version Bump**: Determines `major.minor.patch` based on conventional commits
4. **Changelog**: Generates CHANGELOG.md
5. **Release Notes**: Creates GitHub Release
6. **PyPI**: Publishes to PyPI with `poetry publish`

**Timeline**: ~1-2 minutes after merge

### Commit Message Format

Release automation depends on [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types that trigger release**:

- `feat:` → Minor version bump (0.1.0 → 0.2.0)
- `fix:` → Patch version bump (0.1.0 → 0.1.1)
- `refactor:` → Patch version bump
- `perf:` → Patch version bump
- `BREAKING CHANGE` in footer → Major version bump (0.1.0 → 1.0.0)

**Types that don't trigger release**:

- `docs:`
- `test:`
- `chore:`

### Examples

**Feature release**:

```
feat(search): add LLM-based candidate generation

Implements iterative LLM search with feedback loop.
```

**Patch release**:

```
fix(cache): handle concurrent writes correctly

Use WAL mode to ensure transaction isolation.
```

**Major release**:

```
feat(api)!: redesign Orchestrator API

BREAKING CHANGE: Orchestrator.optimize() no longer accepts legacy SearchParams format.
```

## Manual Release (if needed)

To trigger a release manually:

```bash
# Install semantic-release
npm install -g semantic-release \
  @semantic-release/commit-analyzer \
  @semantic-release/release-notes-generator \
  @semantic-release/changelog \
  @semantic-release/git \
  @semantic-release/github \
  conventional-changelog-conventionalcommits

# Configure credentials
export GITHUB_TOKEN=<your-github-token>
export PYPI_TOKEN=<your-pypi-token>

# Run release
semantic-release
```

## Troubleshooting

### Release not triggered

- Check commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- Verify branch is `master` and push to origin (not local branch)
- Review GitHub Actions logs: https://github.com/flipslidersand/forge/actions

### PyPI publish failed

- Verify `PYPI_TOKEN` is set in GitHub Secrets
- Ensure token has "Upload" permission
- Check PyPI package name matches `forge-kernel`

### Rollback a release

If a bad release was published:

1. **GitHub**: Delete the tag and release on GitHub

   ```bash
   git tag -d v0.2.0
   git push origin :v0.2.0
   ```

2. **PyPI**: Remove the version (requires PyPI admin)
   - Go to https://pypi.org/project/forge-kernel/
   - Select version and delete

3. **Local**: Reset to previous commit
   ```bash
   git reset --hard <previous-commit>
   git push --force origin master  # Only if no one else has fetched
   ```

## CI/CD Pipeline Status

Release workflow: `.github/workflows/release.yml`

**Runs on**: Every push to master
**Duration**: ~1-2 minutes
**Logs**: GitHub Actions → Workflows → Release

## References

- [semantic-release documentation](https://semantic-release.gitbook.io/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Poetry publish](https://python-poetry.org/docs/repositories/#publishing)
