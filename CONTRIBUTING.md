# Contributing to CardioSense ICU Suite

Thank you for contributing to open-source clinical artificial intelligence. To ensure hospital-level safety, all code changes must adhere to strict guidelines.

## Development Workflow
1. **Branch Naming**:
   - `feat/feature-name` (New clinical or UI features)
   - `fix/bug-name` (Clinical math or execution fixes)
   - `chore/bot-task` (Automated workflows and dependency patches)
2. **Commit Conventions**:
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(ml): integrate dynamic recourse optimization`
   - `fix(telemetry): resolve Asystole zero-vital schema validation`
   - `chore(deps): bump fastapi to 0.115`

## Multi-Bot & CI Compliance
Every Pull Request automatically triggers our multi-agent bot quorum:
- **CodeQL**: Deep AST static code security analysis.
- **CodeRabbit AI**: Automated clinical architecture code reviews.
- **Google Scorecard**: Supply-chain dependency integrity checks.
- **Flake8 / ESLint**: PEP8 and TypeScript type safety.

All bots must return green status checks prior to maintainer review and merge.
