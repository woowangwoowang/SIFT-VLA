.PHONY: test check

test:
	CUDA_VISIBLE_DEVICES='' USE_TF=0 HF_HUB_OFFLINE=1 PYTHONPATH=transformers/src:. python -m unittest discover -s tests -v

check:
	python -m compileall -q rd_token experiments prismatic/extern/hf tests transformers/src/transformers/models/llama/modeling_llama.py
	@for script in run_eval/*.sh; do bash -n "$$script" || exit 1; done
