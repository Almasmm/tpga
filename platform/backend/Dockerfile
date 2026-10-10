FROM python:3.12-slim
WORKDIR /srv
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN pip install --no-cache-dir /src fastapi "uvicorn[standard]"
COPY platform/backend/app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
