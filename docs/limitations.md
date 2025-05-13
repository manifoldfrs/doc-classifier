# HeronAI Document Classifier: Limitations, Future Work & Productionization

This document outlines the current limitations of the HeronAI document classification service, proposes extensions to address them, and discusses key considerations for deploying this service in a production environment. It serves as both a transparent assessment of the system's constraints and a roadmap for future enhancements and deployment strategies.

## Table of Contents

- [Current Limitations](#current-limitations)
  - [Classification Accuracy](#classification-accuracy)
  - [Robustness](#robustness)
  - [Scalability](#scalability)
  - [Feature Completeness](#feature-completeness)
  - [Security](#security)
- [Proposed Extensions](#proposed-extensions)
  - [ML Enhancements](#ml-enhancements)
  - [Architecture Improvements](#architecture-improvements)
  - [Feature Additions](#feature-additions)
  - [Security Enhancements](#security-enhancements)
- [Productionization Considerations](#productionization-considerations)
  - [Conceptual Production Architecture Diagram](#conceptual-production-architecture-diagram)
  - [Cloud Deployment (GCP/AWS using Terraform)](#cloud-deployment-gcpaws-using-terraform)
  - [Robustness & Reliability](#robustness--reliability)
  - [Scalability](#scalability-1)
  - [Security](#security-1)
  - [CI/CD for Production](#cicd-for-production)
  - [Data Management & Persistence](#data-management--persistence)
  - [Cost Optimization](#cost-optimization)
  - [Interoperability & Accessibility](#interoperability--accessibility)

## Current Limitations

### Classification Accuracy

1. **Limited Training Data**: The current synthetic data generator produces a relatively small corpus (~3,000 samples) with simplistic patterns that may not reflect the complexity and variety of real-world documents.

2. **Naive Bayes Limitations**: While efficient, the Multinomial Naive Bayes model assumes feature independence, which doesn't hold for text data where word co-occurrences matter.

3. **Heuristic Fallbacks**: The regex-based stage fallbacks provide decent coverage but are brittle when facing documents with ambiguous or unexpected content patterns.

4. **OCR Quality**: The base Tesseract implementation works well for clean images but degrades with poor quality scans, skewed text, or unusual fonts. Preprocessing (deskewing, noise reduction) could improve this.

5. **No Language Awareness**: The current implementation assumes English text and doesn't account for multi-language documents or non-Latin scripts.

### Robustness

1. **Error Handling Gaps**: While the system has structured error handling, certain edge cases (like corrupted files that pass initial validation but fail during parsing) might cause unhandled exceptions or incomplete processing.

2. **Limited Retry Logic**: The current retry mechanism is simulated rather than fully implemented for all failure modes (e.g., transient network issues during OCR calls if external).

3. **No Real Circuit Breakers**: The system lacks protection against cascading failures if dependent services (e.g., a future external metadata service) degrade.

4. **In-Memory Job Storage**: Async jobs are stored in memory, making them vulnerable to service restarts or scaling events. A persistent store (like Redis or a database) is needed.

### Scalability

1. **Single-Process Limitation**: The current design operates within a single process, limiting throughput to the capacity of one server, especially for CPU-bound tasks like OCR or complex ML inference.

2. **No Persistent Queue**: Large batch processing uses in-memory background tasks rather than a durable message queue (like RabbitMQ, SQS, Pub/Sub), limiting reliability and scalability.

3. **Potential Memory Issues**: While streaming is attempted, certain parsing libraries might still load significant data into memory, limiting scalability for very large or complex files.

4. **Request-Response Coupling**: Synchronous classification ties up server resources for the entire processing duration, limiting concurrency.

### Feature Completeness

1. **Limited File Types**: The service currently supports text extraction for **PDF, DOCX, CSV, TXT, and raster images (JPG/JPEG/PNG)**. Other extensions accepted at upload (`doc`, `xls`, `xlsx`, `xlsb`, `md`, `xml`, `json`, `html`, `eml`) lack specific parsers and fall back, yielding low-confidence results.

2. **No Content Extraction**: The classifier identifies document types but doesn't extract structured data fields (e.g., invoice amounts, bank statement transactions, form fields).

3. **Single Label Classification**: Documents receive only one label. Multi-label or hierarchical classification isn't supported.

4. **No Continuous Learning**: The model is static after training. There's no mechanism to improve from user feedback or corrections.

### Security

1. **Basic Authentication**: The API key approach lacks features like rotation, scoping, rate limiting per key, or short-lived credentials.

2. **No PII Detection/Redaction**: The system doesn't identify or redact sensitive information (PII/PHI) in processed documents.

3. **Demo-Grade Security**: Several production security measures (e.g., file encryption at rest, comprehensive audit logging) are documented but not implemented.

## Proposed Extensions

### ML Enhancements

1. **Advanced Models**: Replace/augment Naive Bayes with transformer-based models (e.g., LayoutLM, BERT, DistilBERT) for improved understanding of document context and structure.

2. **Transfer Learning**: Leverage pre-trained document understanding models fine-tuned on specific classification tasks and domains.

3. **Active Learning Pipeline**: Implement a feedback loop where low-confidence classifications are flagged for human review, with corrections feeding back into model retraining.

4. **Ensemble Approaches**: Combine multiple specialist models (e.g., layout-aware, text-based, metadata-based) for more robust classification.

5. **Multi-language Support**: Integrate language detection and use appropriate language-specific models or multilingual models.

6. **Improved OCR**: Integrate OCR preprocessing steps (deskewing, denoising, binarization) and potentially explore alternative OCR engines or cloud-based OCR services for higher accuracy.

### Architecture Improvements

1. **Distributed Processing**:

   - Implement a durable message queue (e.g., RabbitMQ, Redis Streams, SQS, Pub/Sub) for reliable asynchronous processing.
   - Decouple API ingestion from processing using worker services (e.g., Celery, Argo Workflows, KEDA-scaled jobs).
   - Implement a coordinator service or state machine to manage job distribution and status tracking.

2. **Streaming Architecture**:

   - Ensure end-to-end streaming processing, especially for large files, potentially using chunked uploads directly to object storage (S3/GCS) via presigned URLs.
   - Adapt parsing libraries or strategies to work on streamed data where possible.

3. **Caching Layer**:

   - Add Redis caching for duplicate document detection based on content hash (SHA256).
   - Cache classification results for identical content hashes to avoid redundant processing.
   - Consider caching ML model predictions for identical text inputs if beneficial.

4. **Resilience Patterns**:
   - Implement proper circuit breakers (e.g., using `pybreaker` or framework integrations) for external dependencies (future databases, APIs).
   - Implement robust exponential backoff with jitter for all retryable operations (network calls, transient processing failures).
   - Add comprehensive health checking (liveness, readiness probes) for worker nodes and dependent services.

### Feature Additions

1. **Document Understanding**:

   - Extend the pipeline to extract structured data (key-value pairs, tables) from classified documents using techniques like layout analysis, NER, or specialized models.
   - Add entity recognition for key fields (dates, amounts, names, addresses, account numbers).
   - Implement document segmentation to identify functional blocks (header, footer, tables, paragraphs) in complex layouts.

2. **Multi-label & Hierarchical Classification**:

   - Allow documents to be assigned multiple relevant labels (e.g., "invoice" and "contract").
   - Implement hierarchical classification (e.g., financial→statement→bank).
   - Provide confidence scores per category rather than a single overall score.

3. **API Enhancements**:

   - Add a bulk document comparison endpoint (e.g., find duplicates).
   - Implement search capabilities across processed document metadata or extracted text.
   - Add webhook notifications for asynchronous job completion.

4. **Interactive Training UI**:
   - Build an admin interface (e.g., using React/Vue) for reviewing low-confidence classifications.
   - Add tools for annotating new training examples or correcting misclassifications.
   - Provide model performance dashboards and monitoring tools.

### Security Enhancements

1. **Advanced Authentication/Authorization**:

   - Implement JWT-based authentication with short-lived access/refresh tokens.
   - Add OAuth2/OIDC support for integration with identity providers.
   - Implement Role-Based Access Control (RBAC) for different API operations (upload, read, admin).
   - Introduce scoped API keys with specific permissions.

2. **Document Security**:

   - Implement client-side encryption or server-side encryption (SSE) for documents stored at rest (e.g., in S3/GCS).
   - Integrate PII detection and automatic redaction capabilities (using libraries like `presidio` or cloud services like AWS Comprehend/GCP DLP).
   - Support document watermarking for tracking provenance.

3. **Compliance & Auditing**:

   - Implement comprehensive audit logging for all document access and processing operations, stored securely.
   - Define and enforce data retention policies and automatic purging mechanisms.
   - Add features to generate compliance reports (e.g., for GDPR, HIPAA if applicable).

## Productionization Considerations

Moving the HeronAI Document Classifier from a demo/prototype to a robust, scalable, and secure production system requires careful planning across several dimensions. This section outlines key considerations based on the project's tech stack (Python/FastAPI, potentially React/TypeScript) and common cloud deployment patterns (GCP/AWS with Terraform).

### Conceptual Production Architecture Diagram

```mermaid
graph LR
    subgraph "User/Client Realm"
        U[Client Application / UI]
        API_GW[API Gateway e.g., AWS API GW, GCP API GW]
    end

    subgraph "Cloud Environment (GCP/AWS)"
        LB[Load Balancer]
        subgraph "API Service (Stateless)"
            direction LR
            API1[FastAPI Instance 1]
            API2[FastAPI Instance 2]
            API3[FastAPI Instance ...]
        end

        MQ[Message Queue e.g., SQS, Pub/Sub, Redis Streams]

        subgraph "Classifier Workers (Stateless/Stateful)"
            direction LR
            W1[Worker Instance 1]
            W2[Worker Instance 2]
            W3[Worker Instance ...]
        end

        OS[Object Storage e.g., S3, GCS for Files]
        DB[(Database e.g., Postgres/RDS/Cloud SQL for Metadata/Jobs)]
        Cache[(Cache e.g., Redis/ElastiCache for Job State/Dup Check)]
        SecMgr[(Secrets Manager e.g., AWS SM, GCP SM, Vault)]
        Mon[Monitoring & Logging e.g., CloudWatch, Stackdriver, Datadog]
    end

    U --> API_GW;
    API_GW --> LB;
    LB --> API1;
    LB --> API2;
    LB --> API3;

    API1 -- Enqueue Job --> MQ;
    API2 -- Enqueue Job --> MQ;
    API3 -- Enqueue Job --> MQ;

    MQ -- Dequeue Task --> W1;
    MQ -- Dequeue Task --> W2;
    MQ -- Dequeue Task --> W3;

    API1 -- Read Job Status --> DB;
    API2 -- Read Job Status --> DB;
    API3 -- Read Job Status --> DB;

    W1 -- Store File --> OS;
    W2 -- Store File --> OS;
    W3 -- Store File --> OS;

    W1 -- Read/Write Job State/Metadata --> DB;
    W2 -- Read/Write Job State/Metadata --> DB;
    W3 -- Read/Write Job State/Metadata --> DB;

    W1 -- Use Cache --> Cache;
    W2 -- Use Cache --> Cache;
    W3 -- Use Cache --> Cache;

    W1 -- Access Secrets --> SecMgr;
    W2 -- Access Secrets --> SecMgr;
    W3 -- Access Secrets --> SecMgr;
    API1 -- Access Secrets --> SecMgr;

    API1 -- Emit Logs/Metrics --> Mon;
    API2 -- Emit Logs/Metrics --> Mon;
    API3 -- Emit Logs/Metrics --> Mon;
    W1 -- Emit Logs/Metrics --> Mon;
    W2 -- Emit Logs/Metrics --> Mon;
    W3 -- Emit Logs/Metrics --> Mon;

    %% Styling
    classDef api fill:#c9f,stroke:#333,stroke-width:2px;
    classDef worker fill:#f9c,stroke:#333,stroke-width:2px;
    classDef infra fill:#ccf,stroke:#333,stroke-width:1px;
    class API1,API2,API3 api;
    class W1,W2,W3 worker;
    class MQ,OS,DB,Cache,SecMgr,Mon,API_GW,LB infra;

```

**Diagram Explanation:**

- **Clients** interact via an **API Gateway**, providing security, rate limiting, and routing.
- A **Load Balancer** distributes traffic across multiple instances of the stateless **FastAPI API Service**.
- The API service validates requests, potentially handles small synchronous tasks, and **enqueues** larger classification jobs onto a **Message Queue**. It can also query the **Database** for job status.
- **Classifier Workers** pull tasks from the queue. They perform the heavy lifting (parsing, OCR, ML inference).
- Workers interact with **Object Storage** (for raw files if persisted), the **Database** (for job metadata, results), a **Cache** (for job state, deduplication), and **Secrets Manager**.
- Both API and Worker services emit **Logs and Metrics** to a centralized monitoring system.
- The entire infrastructure within the cloud environment should be managed via **Terraform**.

### Cloud Deployment (GCP/AWS using Terraform)

- **Infrastructure as Code (IaC):** Use Terraform to define and manage all cloud resources (compute, storage, networking, queues, databases, IAM roles, etc.). This ensures consistency, repeatability, and version control for the infrastructure. Structure Terraform code using modules for reusability (e.g., VPC module, database module, compute module).
- **Managed Services:** Prefer managed services to reduce operational overhead:
  - **Compute:** GKE/EKS (Kubernetes), Cloud Run/App Runner (Serverless Containers), or EC2/Compute Engine with Auto Scaling Groups. Choose based on control vs. management trade-offs. Kubernetes offers flexibility but higher complexity; serverless containers are simpler for stateless web apps.
  - **Storage:** S3/GCS for durable, scalable file storage. Use lifecycle policies for cost management.
  - **Queues:** SQS/Pub/Sub for reliable, scalable message queuing. Choose standard or FIFO based on ordering requirements.
  - **Database:** RDS/Cloud SQL (Postgres) for relational metadata, job tracking, audit logs. Consider read replicas for scaling read load.
  - **Cache:** ElastiCache/Memorystore (Redis) for caching job states, session data, or duplicate hashes.
- **Networking:** Define a Virtual Private Cloud (VPC) with public and private subnets. Place databases and sensitive resources in private subnets. Use Security Groups / Firewall Rules to control traffic flow strictly. Deploy the API behind a Load Balancer (ALB/ELB/Cloud Load Balancer).
- **Container Registry:** Use ECR/GCR/Artifact Registry to store Docker images securely.

### Robustness & Reliability

- **Health Checks:** Implement comprehensive liveness and readiness probes for API instances and workers within the container orchestrator (Kubernetes, ECS). Readiness probes should ensure the service is fully initialized (e.g., ML models loaded) before receiving traffic. Liveness probes confirm the service is still running.

  ```python
  # Example: Basic FastAPI Liveness/Readiness Endpoint (in api/health.py)
  from fastapi import APIRouter, Response, status

  router = APIRouter()

  # In a real scenario, readiness might check DB connections, model loading status etc.
  _SERVICE_READY = True # Simplified global state for demo

  @router.get("/v1/health/live", status_code=status.HTTP_204_NO_CONTENT, tags=["Health"])
  async def check_liveness() -> Response:
      """Basic liveness check."""
      return Response(status_code=status.HTTP_204_NO_CONTENT) # Service is running

  @router.get("/v1/health/ready", status_code=status.HTTP_204_NO_CONTENT, tags=["Health"])
  async def check_readiness() -> Response:
      """Checks if the service is ready to accept traffic."""
      if _SERVICE_READY:
          # Check essential dependencies: DB connection, models loaded etc.
          # Example: await check_database_connection()
          return Response(status_code=status.HTTP_204_NO_CONTENT)
      else:
          # Service not ready, return 503 Service Unavailable
          return Response(content="Service not ready", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

  # In main application setup (e.g., api/__init__.py):
  # app.include_router(health_router, prefix="/service-status") # Or similar base path
  ```

- **Distributed Tracing:** Implement distributed tracing (e.g., using Datadog, OpenTelemetry with Jaeger, AWS X-Ray, Google Cloud Trace) to track requests across the API, queue, and workers for easier debugging of performance bottlenecks and errors.

  ```python
  # Example: Conceptual Middleware for Trace ID Propagation (using contextvars)
  import uuid
  import contextvars
  from fastapi import Request, Response
  from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

  trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

  class TraceIdMiddleware(BaseHTTPMiddleware):
      async def dispatch(
          self, request: Request, call_next: RequestResponseEndpoint
      ) -> Response:
          # Get trace ID from header or generate a new one
          trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
          trace_id_var.set(trace_id) # Store in context var for access within the request

          response = await call_next(request)
          response.headers["X-Request-ID"] = trace_id_var.get() # Add to response header
          trace_id_var.set(None) # Reset for next request
          return response

  # In main application setup (e.g., api/__init__.py):
  # from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
  # app.add_middleware(TraceIdMiddleware)
  # FastAPIInstrumentor.instrument_app(app) # Example using OpenTelemetry library
  ```

- **Idempotency:** Design worker tasks and API endpoints to be idempotent where possible, especially for operations involving external state changes, to handle retries safely. This often involves checking for a unique idempotency key provided by the client.

  ```python
  # Example: Conceptual Idempotency Check in FastAPI endpoint
  from fastapi import APIRouter, Request, Response, status, HTTPException, Depends
  from typing import Annotated

  router = APIRouter()

  # Assume a simple in-memory store for processed idempotency keys for demo
  PROCESSED_REQUESTS = set()

  async def check_idempotency_key(idempotency_key: Annotated[str | None, Depends(lambda request: request.headers.get("Idempotency-Key"))]):
      if not idempotency_key:
          raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Idempotency-Key header required")

      if idempotency_key in PROCESSED_REQUESTS:
          # Request already processed, return stored result or success indication
          # In a real system, you might fetch the previous response from a store
          raise HTTPException(status_code=status.HTTP_200_OK, detail="Request already processed") # Or return the actual previous result

      # Mark key as being processed (needs locking in concurrent env)
      # PROCESSED_REQUESTS.add(idempotency_key) # Add after successful processing normally

      return idempotency_key

  @router.post("/v1/submit-job", status_code=status.HTTP_202_ACCEPTED)
  async def submit_job(
      request: Request,
      idempotency_key: Annotated[str, Depends(check_idempotency_key)]
  ):
      # Process the job...
      job_result = {"status": "processing", "job_id": "123"}

      # After successful processing, store the result associated with the key
      # and add the key to processed set permanently
      PROCESSED_REQUESTS.add(idempotency_key)
      # store_result(idempotency_key, job_result)

      return job_result
  ```

- **Graceful Shutdown:** Implement graceful shutdown logic (e.g., using FastAPI's `lifespan` events, `signal` handlers in workers) to finish processing in-flight requests/tasks before terminating during deployments or scaling events.

  ```python
  # Example: FastAPI Lifespan for resource cleanup
  from contextlib import asynccontextmanager
  from fastapi import FastAPI
  # Assume asyncpg_pool is an initialized database connection pool

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # Code to run on startup:
      print("Application startup: Connecting to resources...")
      # await connect_to_database()
      # load_ml_models()
      # Example: Initialize DB pool (replace with your actual pool logic)
      # global asyncpg_pool
      # asyncpg_pool = await asyncpg.create_pool(dsn="postgresql://user:pass@host:port/db")
      print("Resources initialized.")
      yield
      # Code to run on shutdown:
      print("Application shutdown: Cleaning up resources...")
      # if asyncpg_pool:
      #     await asyncpg_pool.close()
      #     print("Database pool closed.")
      # await close_other_connections()
      print("Cleanup finished.")

  # In main application setup (e.g., api/__init__.py):
  # app = FastAPI(lifespan=lifespan)
  ```

- **Incident Response:** Have a clear plan for monitoring, alerting, and responding to production incidents.

### Scalability

- **Auto-Scaling:** Configure auto-scaling for both the API service and the worker pool based on metrics like CPU utilization, memory usage, queue depth (for workers), or request latency. KEDA (Kubernetes Event-Driven Autoscaling) can be useful for scaling workers based on queue length. (Managed primarily via Cloud Provider/Kubernetes configuration, not directly in Python app code).
- **Decoupling:** The asynchronous architecture with a message queue is fundamental for scaling. It allows the API and workers to scale independently based on their specific loads.

  ```python
  # Example: Conceptual Task Publishing to a Queue (e.g., Redis Streams or Pub/Sub)
  import json
  # Assume redis_client is an initialized Redis client (e.g., redis-py async client)
  # Assume pubsub_publisher is an initialized GCP Pub/Sub client

  QUEUE_NAME = "classification_jobs" # Example Redis Stream name or Pub/Sub topic

  async def publish_classification_task(job_details: dict):
      """Publishes a job to the message queue."""
      message_body = json.dumps(job_details).encode('utf-8')

      try:
          # --- Example using Redis Streams (requires redis-py) ---
          # await redis_client.xadd(QUEUE_NAME, {"data": message_body})

          # --- Example using GCP Pub/Sub (requires google-cloud-pubsub) ---
          # future = pubsub_publisher.publish(f"projects/my-project/topics/{QUEUE_NAME}", message_body)
          # message_id = await future # Wait for publish confirmation

          print(f"Published task for job ID: {job_details.get('job_id', 'N/A')}")
          # return message_id # Or stream ID for Redis

      except Exception as e:
          # Handle publishing error (log, potentially retry)
          print(f"Error publishing task: {e}")
          raise

  # Called from an API endpoint after validation:
  # await publish_classification_task({"job_id": "xyz789", "file_uri": "s3://bucket/file.pdf"})
  ```

- **Database Scaling:** Use managed database services with read replicas. Implement connection pooling (e.g., using PgBouncer or SQLAlchemy's built-in pooling) to manage database connections efficiently. Consider sharding or partitioning strategies if metadata grows extremely large.

  ```python
  # Example: Basic Connection Pooling with asyncpg (illustrative)
  import asyncpg

  DB_POOL = None

  async def get_db_connection():
      """Dependency function to get a connection from the pool."""
      global DB_POOL
      if DB_POOL is None:
          # Handle initialization error or ensure pool is created at startup
          raise RuntimeError("Database pool not initialized")
      async with DB_POOL.acquire() as connection:
          yield connection # Provide the connection to the endpoint/caller

  async def initialize_db_pool():
      """Call this during application startup (e.g., in lifespan event)."""
      global DB_POOL
      try:
          DB_POOL = await asyncpg.create_pool(
              user='db_user', password='db_password',
              database='app_db', host='127.0.0.1', port=5432,
              min_size=5,  # Minimum number of connections in the pool
              max_size=20  # Maximum number of connections in the pool
          )
          print("Database connection pool initialized.")
      except Exception as e:
          print(f"Failed to initialize database pool: {e}")
          # Handle error appropriately (e.g., prevent app start)

  async def close_db_pool():
      """Call this during application shutdown (e.g., in lifespan event)."""
      global DB_POOL
      if DB_POOL:
          await DB_POOL.close()
          print("Database connection pool closed.")

  # --- Usage in a FastAPI endpoint ---
  # from fastapi import Depends
  #
  # @router.get("/items/{item_id}")
  # async def read_item(item_id: int, db: asyncpg.Connection = Depends(get_db_connection)):
  #     result = await db.fetchrow("SELECT * FROM items WHERE id = $1", item_id)
  #     return result
  ```

- **Stateless Services:** Design the API and worker services to be as stateless as possible. Store state in external systems (database, cache, queue) to facilitate horizontal scaling. (Architectural principle, not a specific code snippet).
- **CDN:** If serving a frontend or static API documentation, use a Content Delivery Network (CDN) like CloudFront or Cloudflare to cache assets closer to users. (Infrastructure configuration).

### Security

- **IAM Best Practices:** Follow the principle of least privilege when configuring IAM roles/permissions for services and users interacting with cloud resources. Use service accounts/instance profiles instead of embedding credentials. (Cloud configuration, use SDKs like `boto3` or `google-cloud-python` which often automatically pick up credentials from the environment).
- **Secrets Management:** Store all sensitive configuration (API keys, database passwords, signing keys) securely using AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault. Inject secrets into applications at runtime, not in Docker images or environment variables directly visible in the orchestrator. For local development, `python-dotenv` is common.

  ```python
  # Example: Using python-dotenv for local development secrets
  import os
  from dotenv import load_dotenv

  # Load variables from a .env file in the project root (or specified path)
  # Typically called once at application startup
  load_dotenv()

  # Access secrets stored in the .env file via environment variables
  DATABASE_URL = os.getenv("DATABASE_URL")
  API_SECRET_KEY = os.getenv("API_SECRET_KEY")

  # In production, environment variables would be injected by the orchestrator
  # or fetched from a secrets manager service using its SDK.
  # Example conceptual fetch (SDK details vary):
  # import boto3 # Example for AWS
  # secrets_client = boto3.client('secretsmanager')
  # secret_value = secrets_client.get_secret_value(SecretId='prod/database/url')['SecretString']
  # DATABASE_URL = secret_value
  ```

- **Network Security:** Use security groups/firewall rules to restrict traffic between services to only necessary ports and protocols. Place public-facing services behind a Web Application Firewall (WAF) like AWS WAF or Cloud Armor for protection against common web exploits (SQLi, XSS). Implement DDoS protection. (Infrastructure configuration).
- **Dependency Scanning:** Integrate tools like `pip-audit`, Snyk, or GitHub Dependabot into the CI/CD pipeline to scan for vulnerabilities in Python dependencies. Generate and store Software Bill of Materials (SBOM). (CI/CD configuration).
- **Input Validation:** Apply rigorous input validation at the API boundary (already partially done with Pydantic and `validate_file`) to prevent injection attacks or malformed data propagation. FastAPI with Pydantic models inherently provides strong input validation.

  ```python
  # Example: Pydantic models enforce input validation in FastAPI
  from fastapi import FastAPI
  from pydantic import BaseModel, Field, EmailStr

  app = FastAPI()

  class UserInput(BaseModel):
      username: str = Field(..., min_length=3, max_length=50)
      email: EmailStr # Built-in validation for email format
      age: int | None = Field(default=None, ge=18, le=120) # Optional, with range validation

  @app.post("/register/")
  async def register_user(user_data: UserInput):
      # If the request body doesn't match the UserInput model,
      # FastAPI automatically returns a 422 Unprocessable Entity error.
      # No need for manual validation code here for the structure/types.
      print(f"Registering user: {user_data.username}")
      # ... process registration ...
      return {"message": "User registered successfully"}
  ```

- **Encryption:**

  - **In Transit:** Enforce HTTPS for all external communication (API Gateway, Load Balancer). Use TLS for internal communication between services where necessary. (Infrastructure configuration).
  - **At Rest:** Enable encryption for data stored in Object Storage (SSE-S3/SSE-GCS or CMEK) and managed databases. Consider application-level encryption for highly sensitive data within files if required.

  ```python
  # Example: Conceptual Application-Level Encryption using 'cryptography' library
  from cryptography.fernet import Fernet

  # --- Key Management (Crucial! Store keys securely, e.g., Secrets Manager) ---
  # Generate a key ONCE and store it safely:
  # key = Fernet.generate_key()
  # Store 'key' securely! Avoid hardcoding.
  # Load the key from secure storage/environment variable:
  # ENCRYPTION_KEY = os.getenv("APP_ENCRYPTION_KEY").encode() # Must be bytes
  # cipher_suite = Fernet(ENCRYPTION_KEY)

  # --- Assume cipher_suite is initialized with the loaded key ---

  def encrypt_data(data: bytes, cipher_suite: Fernet) -> bytes:
      """Encrypts byte data."""
      if not isinstance(data, bytes):
          raise TypeError("Data must be bytes for encryption")
      return cipher_suite.encrypt(data)

  def decrypt_data(encrypted_data: bytes, cipher_suite: Fernet) -> bytes:
      """Decrypts byte data."""
      if not isinstance(encrypted_data, bytes):
          raise TypeError("Encrypted data must be bytes for decryption")
      try:
          return cipher_suite.decrypt(encrypted_data)
      except Exception as e: # Catch specific cryptography errors like InvalidToken
          print(f"Decryption failed: {e}")
          # Handle error appropriately (e.g., log, return error indicator)
          raise # Or return None, depending on desired behavior

  # --- Usage Example ---
  # sensitive_info = b"Patient ID: 12345, Diagnosis: Confidential"
  # Assume cipher_suite is loaded correctly
  # encrypted_info = encrypt_data(sensitive_info, cipher_suite)
  # print(f"Encrypted: {encrypted_info}")
  # Store encrypted_info in DB or file

  # Later, when retrieving:
  # retrieved_encrypted = get_encrypted_data_from_db(...)
  # decrypted_info = decrypt_data(retrieved_encrypted, cipher_suite)
  # print(f"Decrypted: {decrypted_info.decode()}") # Decode if it was originally string
  ```

- **Auditing & Pen Testing:** Conduct regular security audits and penetration tests to identify vulnerabilities. (Process and tooling, not specific Python code).

### CI/CD for Production

- **Environment Promotion:** Implement separate CI/CD pipelines for different environments (e.g., development, staging, production). Promote artifacts (Docker images, Terraform plans) through environments after successful testing and approvals.
- **Deployment Strategies:** Use strategies like Blue/Green or Canary deployments for production releases to minimize downtime and risk. Automate rollback procedures.
- **Infrastructure Automation:** Integrate Terraform plan/apply steps into the CI/CD pipeline, potentially requiring manual approval for production changes. Use tools like `tfsec` or `checkov` to scan Terraform code for security issues.
- **Testing in Pipeline:** Include automated integration tests and potentially end-to-end tests against a staging environment within the CI/CD pipeline before deploying to production.
- **Secrets in CI/CD:** Use secure methods (like OIDC connectors or temporary credentials) for CI/CD pipelines to access cloud resources or secrets managers, avoiding long-lived static credentials.

### Data Management & Persistence

- **Database Backups:** Configure automated backups for the production database (e.g., RDS/Cloud SQL point-in-time recovery and snapshots). Regularly test the restore process. (Database/Infrastructure configuration).
- **Object Storage Lifecycle:** Define lifecycle policies for files stored in S3/GCS to automatically transition older data to cheaper storage tiers (e.g., Infrequent Access, Glacier/Archive) or delete it after a defined retention period. (Cloud storage configuration).
- **Data Retention:** Establish clear data retention policies for job metadata, classification results, and potentially the raw files, based on business needs and compliance requirements. Implement automated cleanup jobs.

  ```python
  # Example: Conceptual Async Cleanup Task Function
  import asyncio
  import datetime
  # Assume db_pool is an initialized asyncpg pool (or similar DB interface)

  RETENTION_DAYS = 90

  async def run_cleanup_job():
      """Periodically cleans up old job records and associated data."""
      while True:
          print("Running cleanup job...")
          cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)

          try:
              async with db_pool.acquire() as conn:
                  async with conn.transaction():
                      # 1. Find old job records
                      old_jobs = await conn.fetch(
                          "SELECT job_id, file_storage_path FROM classification_jobs WHERE completed_at < $1",
                          cutoff_date
                      )

                      if not old_jobs:
                          print("No old jobs found to clean up.")
                          continue

                      job_ids_to_delete = [job['job_id'] for job in old_jobs]
                      print(f"Found {len(job_ids_to_delete)} jobs older than {cutoff_date} to delete.")

                      # 2. Delete associated files from object storage (implement this logic)
                      # for job in old_jobs:
                      #     if job['file_storage_path']:
                      #         await delete_file_from_s3_gcs(job['file_storage_path'])

                      # 3. Delete job records from the database
                      deleted_count = await conn.execute(
                          "DELETE FROM classification_jobs WHERE job_id = ANY($1::uuid[])",
                          job_ids_to_delete
                      )
                      print(f"Deleted {deleted_count} job records from database.")

          except Exception as e:
              print(f"Error during cleanup job: {e}")
              # Add proper error handling/logging

          # Wait for the next run (e.g., 24 hours)
          await asyncio.sleep(60 * 60 * 24)

  # This function would typically be run as a separate scheduled task/worker
  # or invoked periodically by a scheduler (e.g., APScheduler, Celery Beat, K8s CronJob)
  # Example: asyncio.create_task(run_cleanup_job()) # In a long-running process
  ```

- **Disaster Recovery:** Have a documented disaster recovery plan covering data loss and service unavailability scenarios. (Process and infrastructure).

### Cost Optimization

- **Monitoring:** Use cloud provider cost management tools (Cost Explorer, Budgets) and potentially third-party tools to monitor spending. Tag resources appropriately for cost allocation.
- **Right-Sizing:** Regularly review resource utilization (CPU, memory, database instances) and adjust sizes to match the actual workload, avoiding over-provisioning.
- **Storage Tiers:** Utilize appropriate storage tiers in S3/GCS based on access frequency.
- **Spot Instances:** Consider using Spot Instances (EC2/GCP) for stateless, fault-tolerant worker tasks to significantly reduce compute costs, but implement robust handling for instance termination.
- **Serverless:** Evaluate serverless options (Lambda/Cloud Functions, Cloud Run) where appropriate, as they can be cost-effective for event-driven or variable workloads (pay-per-use).

### Interoperability & Accessibility

- **API Gateway:** Use an API Gateway (AWS API Gateway, Apigee/GCP API Gateway) to manage API keys, enforce rate limiting, handle authentication/authorization uniformly, potentially transform requests/responses, and provide a stable entry point for consumers. (Infrastructure configuration).
- **API Versioning:** Implement a clear API versioning strategy (e.g., `/v1/`, `/v2/` in the URL path) to allow for backward-incompatible changes without breaking existing clients.

  ```python
  # Example: API Versioning with FastAPI Routers
  from fastapi import FastAPI, APIRouter

  app = FastAPI(title="HeronAI Classifier")

  # --- Version 1 Router ---
  router_v1 = APIRouter(prefix="/v1", tags=["Version 1"])

  @router_v1.post("/files")
  async def upload_file_v1(file: bytes): # Simplified example
      # ... implementation for v1 ...
      return {"version": "v1", "message": "File processed"}

  # --- Version 2 Router (Example: maybe changes response structure) ---
  router_v2 = APIRouter(prefix="/v2", tags=["Version 2"])

  @router_v2.post("/documents") # Example: endpoint path changed
  async def upload_document_v2(doc: bytes): # Example: parameter name changed
      # ... implementation for v2 ...
      return {"api_version": "v2", "status": "success", "doc_id": "xyz"}

  # Include routers in the main app
  app.include_router(router_v1)
  app.include_router(router_v2)

  # OpenAPI documentation will clearly show endpoints grouped by version/tags.
  ```

- **Documentation:** Maintain up-to-date, comprehensive API documentation (OpenAPI spec generated by FastAPI is a good start) accessible to consumers. Consider hosting a dedicated developer portal. (FastAPI auto-generates `/docs` and `/redoc`).
- **Frontend Integration (React/TypeScript on Vercel):**

  - **CORS:** Configure Cross-Origin Resource Sharing (CORS) correctly on the FastAPI backend (or API Gateway) to allow requests from the Vercel frontend domain.

    ```python
    # Example: Configuring CORS Middleware in FastAPI
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()

    # Define allowed origins (replace with your Vercel frontend URL in production)
    origins = [
        "http://localhost:3000",  # Allow local React dev server
        "https://your-vercel-frontend-app.vercel.app", # Your production frontend
        # Add other origins as needed
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # List of allowed origins
        allow_credentials=True, # Allow cookies to be included in requests
        allow_methods=["*"],    # Allow all standard methods (GET, POST, etc.)
        allow_headers=["*"],    # Allow all headers
    )

    # ... rest of your FastAPI app setup and routes ...

    @app.get("/")
    async def root():
        return {"message": "API is running"}
    ```

  - **Authentication:** Determine how the frontend will authenticate with the backend API (e.g., passing the API key securely, using JWT tokens obtained via an auth flow). Avoid exposing static API keys directly in frontend code. (Implementation depends heavily on the chosen auth method).
  - **API Client:** Use a robust data fetching library in React (like React Query or SWR) to handle API requests, caching, state management, and error handling effectively. (Frontend implementation detail).
