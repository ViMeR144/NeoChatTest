# Multi-stage Dockerfile for Rust Service
FROM rust:1.82 as builder

# Set working directory
WORKDIR /usr/src/app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build the application
RUN cargo build --release

# Production stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /usr/src/app/target/release/neural-network-rust /app/

# Create non-root user
RUN groupadd -r neural && useradd -r -g neural neural
RUN chown -R neural:neural /app

# Switch to non-root user
USER neural

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Start the application
CMD ["./neural-network-rust"]

