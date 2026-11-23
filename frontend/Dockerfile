# =============================================================================
# VEGA FRONTEND - Multi-stage Production Dockerfile
# =============================================================================
# Stages:
# 1. builder - Build React + Vite + R3F application
# 2. production - Nginx serving static files
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: BUILDER
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production=false

# Copy source code
COPY . .

# Build application
ARG VITE_API_URL=http://localhost:8000
ARG VITE_WS_URL=ws://localhost:8000
ARG VITE_APP_NAME=VEGA

ENV VITE_API_URL=${VITE_API_URL}
ENV VITE_WS_URL=${VITE_WS_URL}
ENV VITE_APP_NAME=${VITE_APP_NAME}

RUN npm run build

# -----------------------------------------------------------------------------
# STAGE 2: PRODUCTION
# -----------------------------------------------------------------------------
FROM nginx:alpine AS production

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create nginx user and set permissions
RUN adduser -D -H -u 101 nginx && \
    chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chown -R nginx:nginx /etc/nginx/conf.d

# Switch to non-root user
USER nginx

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# -----------------------------------------------------------------------------
# STAGE 3: DEVELOPMENT
# -----------------------------------------------------------------------------
FROM node:20-alpine AS development

WORKDIR /app

# Install all dependencies including dev
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]