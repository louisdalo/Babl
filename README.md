English | [Français](./README_fr.md)


# Babl
Babl is an app designed to be a book recommendation tool that runs locally.

The initial goal was to rank books that might match one's tastes in [Babelio's Masse Critique events](https://www.babelio.com/massecritique.php) but... I got a little involved and it turned out to be a much larger tool that can be useful to many, so I'm making it public!

![app screenshot](./static/images/screenshot.png)

🐛 [Bug reports & issues](https://github.com/louisdalo/Babl/issues)  

## Features

- 50,000 pre-processed books that can be annotated/recommended
- An annotation tool to indicate which books you like/dislike
- A recommendation tool for books in [Open Library](https://openlibrary.org/)
- Semi-automatic fetch of books in [Babelio's Masse Critique event](https://www.babelio.com/massecritique.php)
- A recommendation tool for books in [Babelio's Masse Critique events](https://www.babelio.com/massecritique.php)
- A Machine Learning model that runs locally, and that you can understand
- Dark and light mode (dark mode is default obviously)

## Getting Started
### Requirements

To use this project you must first install the following softwares on your machine:

- `git`: see [this tutorial](https://github.com/git-guides/install-git)
- `uv`: see [this other tutorial](https://docs.astral.sh/uv/getting-started/installation/)


### Installation

Open a terminal on your computer (type "powershell" in start menu for windows). In the terminal, paste and execute the following commands to copy this project on your machine:

```bash
git clone https://github.com/louisdalo/Babl.git
cd Babl
```
Then, in the same terminal, run the following command to launch the app:

```bash
uv run streamlit run app.py
```

Your browser should open automatically on the app, you are ready to go!

### Tips

All your data lives in the `my_infos` subfolder.

If you want to edit/add/remove informations, this can all be done inside this subfolder. Deleting any file/folder in this subfolder *should* be safe, it will reset the deleted file when you launch the app on your next session.

## Advanced usage
If 50,000 books is not enough for you, then go to [openlibrary.org/developers/dumps](https://openlibrary.org/developers/dumps) and download the "all types" and "ratings" dumps (warning: they are several Go in size). Then, you can use `prepare_data.py` to generate your own dataset for this app.

Run this command for more details:
```bash
uv python prepare.py -h
```

---

## ROADMAP
This lists known issues that I am currently working on, or features that I would like to add in the future:
- [ ] Create an interface to edit labels.
- [ ] French translation
    - [X] README.md
    - [ ] app UI
    - [ ] books metadata
- [ ] Better series grouping (unreliable because based on community tags):
    - [ ] manually detect Erin Hunter "warrior" series.

## Contact

Feel free to get in touch with any issues or suggestions for new features you'd like to see.

- 🐛 **Bug reports & issues:** Use [GitHub Issues](https://github.com/louisdalo/Babl/issues)

For anything else, contact me on [Bluesky](https://bsky.app/profile/louis-dalo.bsky.social)!

---

## Disclaimer

A significant proportion of this code was written with [claude.ai](https://claude.ai), [Anthropic](https://www.anthropic.com/)'s LLM (the free version though, because I don't have that kind of money). I systematically reviewed any code produced by [claude.ai](https://claude.ai), understood it, and modified it when necessary. I believe that it is good practice to to work this way, as **LLMs are just sources of informations, that cannot be blindly trusted.** 

***I can only recommend that everyone do the same.***