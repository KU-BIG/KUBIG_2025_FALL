# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY web/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code (public/ already contains data from local build:data)
COPY web ./

# Build Next.js static export (skip build:data, already done locally)
RUN npx next build

# Stage 2: Production (nginx for static files)
FROM nginx:alpine

# Copy built static files to nginx
COPY --from=builder /app/out /usr/share/nginx/html

# Custom nginx config for SPA routing
RUN echo 'server { \
    listen 80; \
    root /usr/share/nginx/html; \
    index index.html; \
    gzip on; \
    gzip_types text/plain text/css application/javascript application/json; \
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|wav)$ { \
        expires 1y; \
        add_header Cache-Control "public, immutable"; \
    } \
    location / { \
        try_files $uri $uri.html $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
