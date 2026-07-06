# Scoring API — AWS Lambda container image (SPEC Section 5).
# Build AFTER the serving bundle exists:
#   uv run python -m serving.artifact          # writes artifacts/serving/bundle.joblib
#   docker build -t fraud-scoring .
FROM public.ecr.aws/lambda/python:3.12

# Dependencies first (cached across code changes).
COPY serving/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code + model bundle. LAMBDA_TASK_ROOT is on sys.path.
COPY pipeline/ ${LAMBDA_TASK_ROOT}/pipeline/
COPY serving/ ${LAMBDA_TASK_ROOT}/serving/
COPY artifacts/serving/ ${LAMBDA_TASK_ROOT}/artifacts/serving/

ENV MODEL_DIR=${LAMBDA_TASK_ROOT}/artifacts/serving

# API Gateway proxy -> serving.handler.handler(event, context)
CMD ["serving.handler.handler"]
