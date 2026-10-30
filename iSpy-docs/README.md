# iSpy FRC documentation site

This folder contains the Docusaurus site for iSpy. The Python runtime is in the parent repository; run these commands from `iSpy-Docs/`, not from the repository root.

## Run locally

Use Node.js 20 or newer, then install the exact dependencies from the lockfile and start the development server:

```powershell
npm ci
npm start
```

Open `http://localhost:3000/`. Most page and style changes reload in the browser. Stop the server with `Ctrl+C`.

## Edit the docs

- Project documentation pages live in `docs/` as Markdown or MDX.
- The sidebar order and categories are in `sidebars.js`.
- The home page is `src/pages/index.js`; its page-specific styles are in `src/pages/index.module.css`.
- Shared site styling is in `src/css/custom.css`.
- Files placed in `static/` are served from the site root. For example, `static/img/robot.png` is referenced as `img/robot.png` in Docusaurus config.

Use relative doc links such as `./calibration` between pages. Docusaurus applies the correct local or deployed base path for internal links.

## Check a change

Build the site from this directory:

```powershell
npm run build
```

The output is written to `build/`. A successful production build also checks internal links because `onBrokenLinks` is set to `throw`.

## Deployment

The configured production site is hosted on GitHub Pages under `/iSpy-FRC/`. Local development uses `/`. Deployment is normally handled by the repository's GitHub workflow; do not run `npm run deploy` unless you are intentionally using the Docusaurus CLI deployment flow for this repository.
