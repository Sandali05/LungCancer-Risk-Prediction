# syntax=docker/dockerfile:1

FROM node:20-bullseye-slim AS deps
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 \
    NEXT_FONT_IGNORE_FETCH_ERRORS=1
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-bullseye-slim AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 \
    NEXT_FONT_IGNORE_FETCH_ERRORS=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN rm -f next.config.ts
RUN npm run build

FROM node:20-bullseye-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_FONT_IGNORE_FETCH_ERRORS=1
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./next.config.js
RUN chown -R node:node /app
USER node
EXPOSE 3000
CMD ["npm", "run", "start"]
