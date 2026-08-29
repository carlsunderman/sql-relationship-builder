# SQL Relationship Builder - Streamlit container image
#
# Includes the Microsoft ODBC Driver 18 for SQL Server, which pyodbc (src/db.py)
# requires at runtime. Built on the official uv image for fast, reproducible
# dependency installs from uv.lock.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# --- System dependencies: Microsoft ODBC Driver 18 for SQL Server ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg ca-certificates \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
         -o /etc/apt/trusted.gpg.d/microsoft.asc \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list \
         -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer; only re-runs when lockfiles change) ---
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Application code ---
COPY . .

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8550

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default: run the Streamlit app.
# Override with: docker run --entrypoint streamlit <image> run admin/admin.py ...
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
