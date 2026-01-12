# Medical Tutor

A multimodal Socratic medical tutor built on LangGraph.

[Demo video] https://drive.google.com/file/d/1UdihdepyYgDsWbvpIC3nq53O1fcmKn_n/view?usp=drive_link

This system reads medical images, retrieves supporting knowledge (text + image), uses visual tools to gather evidence, teaches Socratically with calibrated probes and hints, and produces a final answer only when the student is ready.

### What “stochastic” means here

The system is **stochastic at the decision points**: the controller samples the next action (tool call vs tutoring action vs answer). The **runtime is deterministic** given a chosen action: routing, tool execution, gating rules, state updates, and trace logging follow fixed logic.

### Architecture (high level)

```text
┌───────────────────────────────────────────────────────────────────────┐
│                          ONLINE RUNTIME (LangGraph)                    │
│                                                                       │
│   pre-retrieve → decide → (tool_exec → decide)* → respond / answer     │
│                                                                       │
│   - pre-retrieve: gather multimodal retrieval context                 │
│   - decide: controller picks next action                              │
│   - tool_exec: executes exactly one tool and persists artifacts        │
│   - respond/answer: tutoring response or final answer                 │
└───────────────────────────────────────────────────────────────────────┘
                    ▲                          ▲
                    │ traces + artifacts       │ playbook rules
                    │                          │
┌───────────────────────────────────────────────────────────────────────┐
│                        OFFLINE IMPROVEMENT LOOPS                        │
│   SFT (action formatting)   RL (tool-use policy)   ACE (playbooks)      │
└───────────────────────────────────────────────────────────────────────┘
```

### Key components

- **Runtime**: LangGraph state machine with explicit shared state and deterministic routing.
- **Controller**: chooses actions (tool calls, tutoring actions, answer) via OpenAI API.
- **Solver**: MedGemma vision-language model for grounded answer generation.
- **Tools**: zoom, enhance, segment (MedSAM2), OCR, retrieval, web, image_findings.
- **Gates**: reveal + assessment gates enforce tutoring constraints.
- **Observability**: per-step traces, tool cards, persisted artifacts.
- **UI**: Gradio app with Tutor + Debug views.

---

## Quick Start

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Set API key
export OPENAI_API_KEY=...

# Check configuration
medical-tutor config
```

Example output:

```text
orchestrator_model=gpt-4o-mini
answer_model=google/medgemma-4b-it
ace_model=gpt-4o-mini
bm25_path=data/retrieval/v3/bm25.pkl
text_index_path=data/retrieval/v3/text_enhanced_new
image_index_path=data/retrieval/v3/images_pmc_vqa1
output_dir=Medical_Tutor/outputs
```

---

## CLI

- `medical-tutor config`  Print configuration summary.
- `medical-tutor run`     Run a one-shot image + question.
- `medical-tutor ui`      Launch the Gradio UI.
- `medical-tutor ace`     Run ACE workflow.
- `medical-tutor eval`    Run evaluation.

---

## Configuration

Configuration is controlled via `MEDTUTOR_` environment variables. See `medical_tutor/config.py` for defaults.

Common settings:

- `MEDTUTOR_ORCHESTRATOR_MODEL`
- `MEDTUTOR_ANSWER_MODEL`
- `MEDTUTOR_ACE_MODEL`
- `MEDTUTOR_BM25_PATH`
- `MEDTUTOR_TEXT_INDEX_PATH`
- `MEDTUTOR_IMAGE_INDEX_PATH`
- `MEDTUTOR_OUTPUT_DIR`

---

## Outputs

Runs write to `output_dir`:

- Per-step traces (state transitions, routing decisions)
- Tool cards (arguments, summaries, errors)
- Persisted artifacts (images, crops, masks, text)
- Retrieval hits and compact summaries

This structure supports debugging, replay, and offline improvement loops (SFT, RL, ACE).

---

## Architecture

For a detailed explanation of the system design, training pipeline, and the decisions behind it, see https://nayanachandrika99.github.io/posts/a-multimodal-stochastic-medical-tutor-a-systems-build-story/
