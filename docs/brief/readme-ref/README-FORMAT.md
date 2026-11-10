# README format (athanor format, black and white only)

Model: https://github.com/ghgoodreau/athanor/blob/main/README.md, also modelled on
https://github.com/matiassingers/awesome-readme.

PALETTE: black and white only. Pure black background, white/grey type and art. No sepia, no colour accents anywhere
(banner, screenshots, diagrams, badges). The live dashboard may be cream and magenta; any README screenshot or GIF of
it must be converted to black and white.

Banner (similar to athanor's):
- 1280x400 PNG at docs/screenshots/banner.png, first line of README: `![name](docs/screenshots/banner.png)`
- Centered project name in a large pixel blackletter/gothic bitmap font, lowercase: shoshin
- Under it, a 1-2 line lowercase monospace tagline: "it has never seen this game. watch it learn."
- Left and right edges: Albrecht Dürer, Melencolia I (1514, public domain; its magic square echoes the game grid),
  dithered to two tones (1-bit/ordered or Floyd-Steinberg), fading into the black. Credit the source in NOTICE.md.

Structure:
1. Banner, then 2-3 plain sentences saying what it is.
2. One large hero screenshot of the real product/UI right away (B&W).
3. Short themed sections: ## The Game, ## The Surprise, ## The Ledger, each a few short plain paragraphs followed by
   one real screenshot or plot. Inline `code` for commands.
4. A 2x2 markdown image grid where it helps.
5. ## Install: one sh block (clone, dry run, run), then what it touches and how to undo. Offline eval needs no key.
   Live play documents the optional free ARC_API_KEY (user's own sign-up at ARC Prize; never committed).
6. ## The Name: shoshin is the Zen idea of beginner's mind, meeting each thing openly, as a beginner would.
7. ## Inspirations: short, links INSPIRATIONS.md / related work (WARC, EATA, closest ARC-AGI-3 world-model papers).
8. ## License: MIT, third-party terms in NOTICE.md.

Voice: calm, literal, short sentences, a little wit, no hype, no emoji.
Under the tagline paragraph: the GitHub topics as hashtags, and a single small row of black-and-white shields.io
badges (style=flat-square, colour black/white). Demo GIF/video from a real run is required.
