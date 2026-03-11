# production_template
[![CI Status](https://github.com/Eduardo1100/production_template/actions/workflows/ci.yml/badge.svg)](https://github.com/Eduardo1100/production_template/actions)

This project is a production-ready AI template using `uv`, `mise`, and `Graphite`.

Database migrations use Alembic.

- Apply migrations: `make db-upgrade`
- Show current revision: `make db-current`
- Generate a new migration: `make db-revision MESSAGE="describe change"`
