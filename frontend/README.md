# Lung Cancer Frontend

This directory contains the Next.js frontend for the lung cancer risk prediction demo. The app is configured to build under strict TypeScript settings and to run well in offline-friendly Docker builds.

## Prerequisites
- Node.js 18 or newer
- npm 9 or newer

## Available scripts
- `npm run dev` — start the development server at [http://localhost:3000](http://localhost:3000)
- `npm run build` — generate an optimized production build (used in CI/Docker)
- `npm start` — run the production build with `next start`
- `npm test` — run the ESLint suite (alias for `npm run lint`)

## Styling notes
The default layout uses system UI fonts so the Docker image does not require network access for font downloads. Global styles live in [`app/globals.css`](app/globals.css) and the top-level layout is defined in [`app/layout.tsx`](app/layout.tsx).

## Docker usage
A multi-stage Dockerfile is provided in [`Dockerfile`](Dockerfile). Build the image with:

```bash
docker build -t lung-frontend .
```

To run the container:

```bash
docker run --rm -p 3000:3000 lung-frontend
```

The container listens on port 3000 by default.
