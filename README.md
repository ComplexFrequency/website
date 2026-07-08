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
