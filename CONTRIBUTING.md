# Contributing to **AI-Resource**

Thank you for taking the time to contribute! 🎉  
This repository aims to be a *free knowledge hub* for Machine-Learning / Deep-Learning learners.  
Please read the short rules below **before** opening a Pull Request (PR).

---

## 1. Repository layout (recap)

```
AI-Resource/
├── offline/               # PDF files stored inside the repo
│   ├── Books/             # textbooks, reference books, theses …
│   └── Papers/            # research papers, slide decks, technical notes
└── online/                # markdown files that only contain external links
    ├── books.md           # free/open-access textbooks
    └── papers/            # one topic ⇒ one markdown file (e.g. computer-vision.md)
```

*Never* commit binaries or assets into `online/` — that area is link-only.

---

## 2. Adding **offline** resources (PDF)

1. Put the PDF into the correct sub-folder of `offline/Books` or `offline/Papers`.
2. Use **Title Case** filenames, separate words with spaces or dashes, keep extension `.pdf`.
3. Update **README.md**: add an entry under the corresponding list (respect alphabetical order if possible).
4. Make sure the link in `README.md` works *after cloning*.

> Example
>
> `offline/Papers/Computer Vision/YOLO/YOLOv7.pdf`

---

## 3. Adding **online** resources (external links)

1. Pick or create the right markdown file inside `online/`.
   * `online/books.md` – textbooks/ebooks
   * `online/papers/<topic>.md` – research papers, blogs
2. Format each item like
   ```md
   - "Paper Title" – *Conference/Journal Year*  
     https://arxiv.org/abs/xxxx.xxxxx
   ```
3. Optional: add a short 1-line note or ⭐ emoji for seminal papers.
4. **Do NOT** add the PDF itself — links only.

---

## 4. Pull Request checklist

- [ ] The resource is open access or its license allows sharing.
- [ ] Filename / markdown entry follows the conventions above.
- [ ] `README.md` updated (offline) **or** proper markdown file updated (online).
- [ ] `scripts/check_links.py` passes (run `python scripts/check_links.py`).
- [ ] PR description briefly states *why* the resource is useful.

---

## 5. Running the link checker (optional but recommended)

```bash
pip install -r scripts/requirements.txt  # only needed once
python scripts/check_links.py
```

The script performs a `HEAD` request to every URL in `online/` and prints broken links.

---

## 6. Code of conduct

Be respectful and constructive. We are all here to learn. 🙌

Happy contributing!
