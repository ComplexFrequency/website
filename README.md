# My own blog

Personal site and blog, built with [Jekyll](https://jekyllrb.com/) and hosted on GitHub Pages at
<https://complexfrequency.github.io/website/>.

## Writing

**New article** — add a Markdown file to `_posts/` named `YYYY-MM-DD-short-title.md`:

```markdown
---
title: My Article Title
description: One sentence shown in the article list and in search results.
---

Article body in Markdown.
```

**New project** — add a Markdown file to `_projects/` (any name):

```markdown
---
title: My Project
description: One sentence shown on the project card.
tags: [Python, ML]
image: /assets/images/my-project.png   # optional card screenshot
link: https://github.com/...           # optional external link
---

Project write-up in Markdown.
```

That's it — push to `main` and GitHub Pages rebuilds the site.

**Math in posts** — set `math: true` in the front matter to load KaTeX on that page:

```markdown
---
title: My Article Title
math: true
---

Inline math like $$e^{i\pi} + 1 = 0$$ works in body text, and so does a display block:

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}
$$
```

kramdown picks inline vs. display based on where the `$$...$$` sits in the block. Inside raw HTML
blocks (such as `<details>` asides), kramdown doesn't process Markdown, so use `\( ... \)` and
`\[ ... \]` there instead of `$$`.

## Structure

| Path | What it is |
|---|---|
| `_config.yml` | Site title, author, links, plugins |
| `_layouts/` | HTML templates (nav, footer, page shells) |
| `assets/css/style.css` | The one stylesheet |
| `index.md` | About page |
| `articles.html`, `projects.html` | List pages |
| `_posts/`, `_projects/` | Content — one Markdown file each |

## Local preview (optional)

Requires a recent Ruby, then:

```sh
bundle install
bundle exec jekyll serve
```

and open <http://localhost:4000/website/>.
