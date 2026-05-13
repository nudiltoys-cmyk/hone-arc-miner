# Colab Lab

Use Colab only as the research lab. Do not put wallet keys, hotkeys, seed phrases, API keys, or TAO account material in a notebook.

## Flow

1. Push this repository to GitHub.
2. Open `notebooks/hone_colab_eval.ipynb` in Google Colab.
3. Set `REPO_URL` to your public GitHub repository.
4. Run the notebook.
5. Share the final exact-match, partial, grid, and elapsed numbers back here.

## LLM Experiment

Open `notebooks/hone_colab_llm_eval.ipynb` to test a local Hugging Face instruct model. The default is small enough for a CPU smoke test. If Colab gives a GPU, switch to `Qwen/Qwen2.5-7B-Instruct` with `LOAD_4BIT=True`.

## Decision Gate

We only move to registration when one of these is true:

- Local/Colab tests show credible exact-match improvement.
- A validator-style vLLM run shows the solver can clear Hone's reward floor.

Until then, we keep spend to almost zero.
