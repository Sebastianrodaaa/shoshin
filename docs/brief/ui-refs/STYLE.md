# shoshin dashboard style

- Background: warm paper cream `#f4eee2`; panels a lighter cream `#fbf8f1` with a 1px ink hairline.
- Ink: `#141210` for type and rules.
- Surprise: one hot magenta `#ff1f8f` for a changed cell the model got wrong. It cools through pink to pale
  lilac `#d9c8f0` once the rule behind it is learned (predicted right again). No other accent colour.
- Game cells are the hero: three equal square panels across the top (real frame, committed prediction,
  surprise map), drawn with the ARC-AGI-3 16-colour palette, crisp pixels, no smoothing.
- Typography: a pixel bitmap face (Silkscreen, OFL, vendored) for numbers and the name; a plain grotesque
  (system sans: Helvetica / Arial) for labels. No serif, no monospace body.
- Layout: top row of three panels + narrow right rail (compute meter: gradient steps vs WARC's every-move recipe);
  full-width rule ledger strip underneath that ticks "rule 4 learned after 3 tries".
- Motion: on load the whole board flares magenta within the first second; patches go quiet as rules are learned.
  Everything readable with the sound off; captions carry the story.
