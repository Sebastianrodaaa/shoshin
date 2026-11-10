PY ?= python3
WARC_REV ?= main
GAMES ?= ft09
SEEDS ?= 0 1 2
T ?= 40

.PHONY: setup games smoke eval repro demo dashboard video test

setup:  ## CPU torch, WARC (CC0) and its fold checkpoints
	test -d third_party/WARC || git clone --depth 1 -b $(WARC_REV) https://github.com/third-intelligence/WARC third_party/WARC
	$(PY) -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r third_party/WARC/requirements.txt -r requirements.txt
	$(PY) -c "from huggingface_hub import snapshot_download as s; s('third-intelligence/WARC', local_dir='models', allow_patterns=['manifest.json', 'fold*/checkpoint_best.pt'])"

games:  ## public game code for play-yourself (anonymous, read-only)
	cd third_party/WARC/agent && $(PY) -c "from warc_eval.prepare import prepare; prepare('$(CURDIR)/games')"

smoke:  ## one held-out game, every arm, one seed, timed
	$(PY) -m shoshin.run --games ft09 --seeds 0 --transitions 20 --out results/smoke

eval:   ## the full plan (see docs/REPORT.md for the measured cost)
	$(PY) -m shoshin.run --games $(GAMES) --seeds $(SEEDS) --transitions $(T)

repro:  ## tables, summary and plots from the cached per-transition logs in results/runs
	$(PY) scripts/plots.py

demo:   ## the traced real run the dashboard replays
	$(PY) -m shoshin.run --games cn04 --arms gated --seeds 0 --transitions 40 --trace --out results/demo

dashboard:
	$(PY) app/server.py

video:  ## 1080p MP4 + README GIF of the real dashboard
	$(PY) scripts/record_demo.py

test:
	$(PY) test_shoshin.py
