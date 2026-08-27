# FastFlowLM Website Content Guide

This guide will help you add and edit content on the FastFlowLM website, even if you're not a developer. The website uses simple text files (Markdown) that you can edit directly on GitHub.

## Table of Contents

- [Getting Started](#getting-started)
- [How to Edit Content on GitHub](#how-to-edit-content-on-github)
- [Understanding Pages and Sections](#understanding-pages-and-sections)
- [Available Section Types](#available-section-types)
  - [Hero Section](#hero-section)
  - [Two Column Section](#two-column-section)
  - [Cards Section](#cards-section)
  - [Media Section](#media-section)
  - [People Section](#people-section)
- [Common Tasks](#common-tasks)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Getting Started

The website content is stored in **Markdown files** (`.md` files) in this repository. Each page on the website corresponds to a Markdown file. For example:
- `index.md` = The homepage
- `team.md` = The team page
- `models.md` = The models page

You don't need to install anything or run code—you can edit these files directly on GitHub!

---

## How to Edit Content on GitHub

### Step 1: Navigate to the File

1. Go to the [GitHub repository](https://github.com/FastFlowLM/landingpage) (or your repository URL)
2. Find the file you want to edit (e.g., `index.md` for the homepage)
3. Click on the file name to open it

### Step 2: Edit the File

1. Click the **pencil icon** (✏️) in the top-right corner of the file view
2. Make your changes in the editor
3. Scroll down to the bottom of the page

### Step 3: Save Your Changes

1. In the "Commit changes" section at the bottom:
   - **Commit message**: Write a brief description of what you changed (e.g., "Updated homepage hero text")
   - Leave the other options as default
2. Click the green **"Commit changes"** button

That's it! Your changes will be saved and the website will update automatically (usually within a few minutes).

---

## Understanding Pages and Sections

Each page file has two parts:

1. **Front Matter** (at the top, between `---` lines): This contains page settings like title, description, and the list of sections
2. **Sections**: The actual content blocks that appear on the page

Here's what a typical page looks like:

```yaml
---
layout: page
title: "Page Title"
permalink: /page-url/
description: "Page description for search engines"
sections:
  - type: hero
    title: "Welcome"
    body: "This is the hero section"
  - type: two_column
    left:
      title: "Left side"
    right:
      title: "Right side"
---
```

The `sections` array is a list of content blocks. Each section has a `type` (like `hero`, `two_column`, `cards`, etc.) and then specific content fields.

---

## Available Section Types

### Hero Section

The hero section is typically used at the top of a page for the main headline and call-to-action buttons.

**When to use:** Homepage banner, page headers, important announcements

**Example:**
```yaml
- type: hero
  id: top
  kicker: "Optional small text above title"
  title: "Main Headline"
  body: |
    This is the main description text.
    You can write multiple lines here.
  ctas:
    - label: "Download Now"
      href: "https://example.com/download"
      style: primary
      external: true
    - label: "Learn More"
      href: "/docs/"
      style: ghost
  stats:
    - label: "Runtime size"
      value: "~17 MB"
    - label: "Context"
      value: "Up to 256k tokens"
```

**Available fields:**
- `kicker` - Small text that appears above the title (optional)
- `title` - Main headline
- `body` - Description text (supports multiple lines with `|`)
- `ctas` - List of buttons (call-to-action)
  - `label` - Button text
  - `href` - Link URL
  - `style` - Either `primary` (highlighted) or `ghost` (outlined)
  - `external` - Set to `true` if linking to an external website
- `stats` - List of statistics/metrics to display
  - `label` - Stat name
  - `value` - Stat value
- `right` - Optional right-side content (can include carousel, code blocks, or other content)

---

### Two Column Section

A section that displays content in two columns (left and right).

**When to use:** Comparing features, showing code examples alongside descriptions, side-by-side content

**Example:**
```yaml
- type: two_column
  id: install
  variant: alt  # Optional: changes styling
  left:
    kicker: "Install"
    title: "Getting Started"
    body: "Instructions go here"
    items:
      - heading: "Step 1"
        body: "Description of step 1"
      - heading: "Step 2"
        body: "Description of step 2"
    ctas:
      - label: "Get Started"
        href: "/docs/"
        style: primary
  right:
    kicker: "Quickstart"
    code_blocks:
      - title: "CLI"
        content: |
          flm run llama3.2:1b
          flm list
    metric_cards:
      - label: "Performance"
        value: "72 tok/s"
        desc: "Ryzen AI 9 HX 370"
```

**Available fields for left/right:**
- `kicker` - Small text above title
- `title` - Section title
- `body` - Description text
- `items` - List of feature items with `heading` and `body`
- `code_blocks` - List of code examples with `title` and `content`
- `metric_cards` - List of metric displays with `label`, `value`, and `desc`
- `ctas` - List of buttons
- `pills` - List of tag-like elements (can be strings or objects with `label` and `href`)

---

### Cards Section

Displays a grid of card components, perfect for showcasing features, models, or categories.

**When to use:** Feature lists, model catalogs, service offerings

**Example:**
```yaml
- type: cards
  id: models
  kicker: "Models"
  title: "Available Models"
  body: "Choose from our curated collection"
  cards:
    - label: "Flagship"
      title: "Llama 3.2"
      body: "Optimized kernels for 70B down to 1B"
      ctas:
        - label: "Learn More"
          href: "/docs/models/llama/"
          style: ghost
    - label: "Vision"
      title: "Gemma 3 VLM"
      body: "Vision and text pipeline on NPU"
  ctas:
    - label: "Browse All Models"
      href: "/models/"
      style: primary
```

**Available fields:**
- `kicker` - Small text above title
- `title` - Section title
- `body` - Description text
- `cards` - List of cards, each with:
  - `label` - Small tag above card title (optional)
  - `title` - Card headline
  - `body` - Card description
  - `href` - Optional link (makes entire card clickable)
  - `ctas` - Buttons inside the card
- `ctas` - Buttons below the card grid

---

### Media Section

Displays an image or media with accompanying text content.

**When to use:** Product demos, screenshots, visual showcases

**Example:**
```yaml
- type: media
  kicker: "Demo"
  title: "See It In Action"
  body: "Watch FastFlowLM in action"
  media:
    src: "/assets/demo.gif"
    alt: "FastFlowLM demo"
    title: "Interactive Demo"
    body: "Experience the power of NPU-first AI"
    items:
      - heading: "Feature 1"
        body: "Description of feature"
      - heading: "Feature 2"
        body: "Description of feature"
```

**Available fields:**
- `kicker`, `title`, `body` - Standard header fields
- `media` - Media object with:
  - `src` - Image path (use `/assets/filename.ext` for images in the assets folder)
  - `alt` - Alternative text for accessibility
  - `kicker`, `title`, `body` - Text content next to the media
  - `items` - List of feature items

---

### People Section

Displays a grid of team member or person cards.

**When to use:** Team pages, contributor lists, advisor showcases

**Example:**
```yaml
- type: people
  kicker: "Team"
  title: "Meet the Team"
  people:
    - name: "John Doe"
      role: "Lead Engineer"
      image: "/assets/john.jpg"
      bio: "John has 10 years of experience in AI systems."
      links:
        - label: "LinkedIn"
          url: "https://linkedin.com/in/johndoe"
        - label: "GitHub"
          url: "https://github.com/johndoe"
    - name: "Jane Smith"
      role: "Product Manager"
      image: "/assets/jane.jpg"
      bio: "Jane leads product strategy and community engagement."
```

**Available fields:**
- `kicker`, `title`, `body` - Standard header fields
- `people` - List of person objects, each with:
  - `name` - Person's name
  - `role` - Job title or role
  - `image` - Photo path (use `/assets/filename.ext`)
  - `bio` - Biography text
  - `links` - List of social/contact links with `label` and `url`

---

## Common Tasks

### Adding a New Page

1. Create a new `.md` file in the root directory (e.g., `about.md`)
2. Add the front matter at the top:
   ```yaml
   ---
   layout: page
   title: "About"
   permalink: /about/
   description: "About FastFlowLM"
   sections:
     - type: hero
       title: "About Us"
   ---
   ```
3. Add your sections below the front matter
4. Save the file

### Editing the Homepage

1. Open `index.md`
2. Find the section you want to edit
3. Modify the text, links, or add/remove sections
4. Save your changes

### Adding Images

1. Upload your image to the `assets/` folder on GitHub
2. Reference it in your content using: `/assets/your-image.jpg`
3. Example in a media section:
   ```yaml
   media:
     src: "/assets/your-image.jpg"
     alt: "Description of image"
   ```

### Changing Button Links

Find the `ctas` array in any section and modify the `href` field:
```yaml
ctas:
  - label: "Button Text"
    href: "/new-page/"  # Change this URL
    style: primary
```

### Reordering Sections

Simply move the section blocks up or down in the `sections` array. The order in the file is the order on the page.

---

## Tips and Best Practices

### Text Formatting

- Use `|` after `body:` to write multiple lines:
  ```yaml
  body: |
    Line 1
    Line 2
    Line 3
  ```

- For single-line text, you can use quotes:
  ```yaml
  body: "Single line of text"
  ```

### Links

- **Internal links** (to other pages on the site): Use paths like `/docs/` or `/models/`
- **External links** (to other websites): Use full URLs like `https://github.com/...` and set `external: true`

### Images

- Keep image file sizes reasonable (under 1MB when possible)
- Use descriptive filenames (e.g., `team-photo.jpg` not `IMG_1234.jpg`)
- Always include `alt` text for accessibility

### Testing Your Changes

1. After saving, wait a few minutes for the site to rebuild
2. Visit the website to see your changes
3. If something looks wrong, check:
   - YAML indentation (use spaces, not tabs)
   - Quotation marks around text with special characters
   - File paths for images and links

### Getting Help

- Check existing pages (like `index.md` or `team.md`) for examples
- Make sure your YAML syntax is correct (proper indentation, matching quotes)
- When in doubt, copy a similar section from another page and modify it

---

## Quick Reference

| Section Type | Best For | Key Fields |
|-------------|----------|------------|
| `hero` | Page headers, main banners | `title`, `body`, `ctas`, `stats` |
| `two_column` | Side-by-side content | `left`, `right` with various sub-fields |
| `cards` | Feature grids, catalogs | `cards` array with `title`, `body` |
| `media` | Demos, screenshots | `media` with `src`, `title`, `body` |
| `people` | Team pages | `people` array with `name`, `role`, `image` |

---

**Remember:** You can always preview your changes on GitHub before they go live, and you can always revert changes if something goes wrong. Happy editing! 🎉

