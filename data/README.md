# Data

shoshin replays the official ARC-AGI-3 Public Demo human recordings (ARC Prize Foundation,
https://arcprize.org/blog/arc-agi-3-human-dataset). It uses WARC's layout and WARC's checksum list
(`data/SHA256SUMS`, copied unchanged from https://github.com/third-intelligence/WARC/blob/main/data/SHA256SUMS):

```text
data/ARC-AGI-3/data/training/<game>/<id>.recording.jsonl
```

`shoshin.replay.recording(game)` picks the first recording of a game by file name and refuses it if its SHA-256
does not match `SHA256SUMS`.

## What is in the repository

Two real recordings, so the tests and `make demo` run without a download:

| game | recording | size | held out by |
| --- | --- | --- | --- |
| ft09 | `01f76fc7-c612-4006-8d08-c13ed0cb91de.recording.jsonl` | 3.4 MB | `fold3` (tests) |
| cn04 | `14a28e82-4f24-4cb6-8810-39d70b9d0b4b.recording.jsonl` | 6.6 MB | `fold2` (dashboard demo) |

## The rest

The full Public Demo set is linked from the dataset post above (a public Google Drive folder). Download it, put the
files in the layout above, then check them:

```sh
cd data && grep -F -f <(find ARC-AGI-3 -name '*.jsonl' | sed 's|^|  |') SHA256SUMS | sha256sum -c --quiet && echo OK
```

The public game code used by "play yourself" comes from WARC's anonymous, read-only download:
`make games` (writes `games/`, no API key, no scorecard).
