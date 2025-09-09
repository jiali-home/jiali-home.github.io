Website maintenance guide

- News: edit `assets/data/news.json`
  - Add an object: `{ "date": "YYYY-MM", "text": "…", "links": [{"href": "https://…", "label": "…"}] }`
  - Items are auto-sorted by `date` (descending).

- Publications: edit `assets/data/publications.json`
  - Fields: `title`, `tag` (e.g., "CVPR 2024"), `authors` (array), `image` (path under `assets/images`), `description`, `link`, `date` (YYYY-MM)
  - Items are auto-sorted by `date` (descending).
  - Put images in `assets/images/` and reference like `./assets/images/your_image.png`.

- Rendering
  - `assets/js/script.js` fetches the JSON files on load and injects items into `index.html`.
  - No build step required. Push changes to GitHub Pages and refresh.

- Tips
  - To emphasize your name in authors, keep it as `Jia Li` (bolded automatically).
  - If JSON fails to load, check browser console and JSON syntax (commas/quotes).
