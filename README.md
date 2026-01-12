# Medical_Tutor (Standalone)

<video src="assets/demo.mov" controls muted playsinline></video>

[Demo video](assets/demo.mov)

This is a standalone **multimodal Socratic medical tutor** built under `Medical_Tutor/` using LangGraph. The runtime is a traceable state machine that can:

- read medical images and describe objective findings,
- retrieve supporting medical knowledge (text + image),
- call tools to gather evidence,
- tutor Socratically with calibrated probes and hints,
- and reveal a final answer only when the student asks.

Core components (brief):

- **Runtime loop**: pre-retrieve → decide → (tool_exec → decide)* → respond/answer
- **Controller**: chooses tutor actions and tool calls (OpenAI Responses API)
- **Solver**: MedGemma answer model (used for non-reveal answers)
- **Tools**: zoom, enhance, segment, OCR, retrieval, web, image_findings
- **Gates**: reveal and assessment gates enforce pedagogy
- **Observability**: per-step traces, tool cards, artifacts, errors
- **UI**: Gradio-based Tutor + Debug Run tabs

## Quick Start

From the repository root:

1) Create/activate a virtual environment (example uses system Python):

    python -m venv .venv
    source .venv/bin/activate

2) Install the package in editable mode:

    python -m pip install -e Medical_Tutor

3) Set required API keys (for tutor/controller + web tools if used):

    export OPENAI_API_KEY=...

4) Print the configuration summary:

    medical-tutor config

Expected output:

    orchestrator_model=gpt-4o-mini
    answer_model=google/medgemma-4b-it
    ace_model=gpt-4o-mini
    bm25_path=data/retrieval/v3/bm25.pkl
    text_index_path=data/retrieval/v3/text_enhanced_new
    image_index_path=data/retrieval/v3/images_pmc_vqa1
    output_dir=Medical_Tutor/outputs

## CLI Commands

- `medical-tutor config`  Print configuration summary.
- `medical-tutor run`     Run a one-shot image+question.
- `medical-tutor ui`      Launch the Gradio UI.
- `medical-tutor ace`     Run ACE workflow.
- `medical-tutor eval`    Run evaluation.

## Configuration

Configuration is controlled via `MEDTUTOR_` environment variables. See `medical_tutor/config.py` for defaults.
