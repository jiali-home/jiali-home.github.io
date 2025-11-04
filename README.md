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

Blog

- Blog list: edit `assets/data/blog.json`
  - Add an object: `{ "title": "…", "date": "YYYY-MM" (or YYYY-MM-DD), "summary": "…", "image": "./assets/images/…", "link": "./blog/your-post.html" }`
  - Items are auto-sorted by `date` (descending).
  - You can also point `link` to an external article (e.g., Medium) instead of a local file.
- New post page (optional):
  - Copy `blog/hello-world.html` to `blog/<your-slug>.html` and edit the title/date/content.
  - Keep the CSS link paths relative (e.g., `../assets/css/style.css`).

Markdown Blog (Jekyll on GitHub Pages)

- Write posts as Markdown files under `_posts/` with filename `YYYY-MM-DD-your-title.md`.
- Each post needs front matter:
  ```
  ---
  layout: post
  title: Your Title
  summary: Optional one-line summary
  ---
  
  Your markdown content here.
  ```
- Blog index is at `/blog/` and lists all posts; each post renders with `_layouts/post.html` using site CSS.
- No build step required locally — GitHub Pages auto-builds Jekyll when you push.
- Nav: Clicking "Blog" on the home page routes to `/blog/`.


### Preview
`bundle exec jekyll serve --livereload`