# LLaMA Factory Codebase Analysis (Generated Summary)

This document summarizes the structure, components, and workflow of the LLaMA Factory codebase based on an analysis session.

## 1. Overall Architecture Overview

LLaMA Factory is a comprehensive framework designed for accessible and efficient fine-tuning of a wide variety of large language models (LLMs). It aims to simplify fine-tuning via command-line (CLI) and a graphical web interface (LLaMA Board), supporting numerous models, training methods, and optimization techniques.

**Key Goals & Features:**

*   Fine-tune 100+ LLMs (LLaMA, Mistral, Qwen, Gemma, etc.).
*   Supports various training methods: (Continuous) Pre-training, Supervised Fine-tuning (SFT), Reward Modeling (RM), PPO, DPO, KTO, ORPO.
*   Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA (various bits), DoRA, LoRA+, PiSSA, etc.
*   Performance Optimizations: FlashAttention-2, Unsloth, RoPE Scaling, Gradient Checkpointing, DeepSpeed ZeRO, etc.
*   Task Support: Dialogue, Tool Use, Multimodal (Image, Video, Audio).
*   Interfaces: CLI, Web UI (Gradio), OpenAI-style API.
*   Monitoring: TensorBoard, WandB, MLflow, SwanLab integration.

## 2. Top-Level Directory Structure

*   `run.py`: PBS/Cluster-specific job submission script (wraps `llamafactory-cli`).
*   `src/`: Main Python source code package (`llamafactory`).
*   `examples/`: Example configurations (YAML) and scripts.
*   `data/`: Default directory for datasets and the main dataset configuration file (`dataset_info.json`).
*   `evaluation/`: Scripts/modules for model evaluation.
*   `requirements.txt`: Core Python dependencies.
*   `README.md`: Original project documentation.
*   `setup.py`, `pyproject.toml`: Packaging files.
*   `scripts/`: Utility scripts (e.g., `vllm_infer.py`).
*   `tests/`: Automated tests.
*   `docker/`: Docker configuration files.
*   `.github/`: CI/CD workflows (GitHub Actions).
*   `src/llamafactory/cli.py`: Main entry point for the `llamafactory-cli` command.
*   `src/llamafactory/launcher.py`: Helper script launched by `torchrun` for distributed training.
*   `src/llamafactory/hparams/`: Configuration dataclasses (`ModelArguments`, `DataArguments`, etc.).
*   `src/llamafactory/model/`: Model loading, patching, adapter logic.
*   `src/llamafactory/data/`: Data loading, processing, formatting pipeline.
*   `src/llamafactory/train/`: Training workflow logic (SFT, DPO, RM, etc.).
*   Other directories (`api/`, `chat/`, `eval/`, `webui/`, `extras/` within `src/llamafactory`): Support specific functionalities.

## 3. Key Dependencies (`requirements.txt`)

*   **Hugging Face Ecosystem:** `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `tokenizers`.
*   **Web UI/API:** `gradio`, `fastapi`, `uvicorn`, `sse-starlette`.
*   **Data Handling:** `pandas`, `numpy`, `pyyaml`.
*   **Optimizers/Schedulers:** Optionally `bitsandbytes`, `galore_torch`, `torch_optimizer`, `badam`.
*   **Tokenization:** `sentencepiece`, `tiktoken`.
*   **Utilities:** `einops`, `protobuf`, `fire`, `tyro`, `packaging`, `pydantic`, `wandb`, `mlflow`, `swanlab`.
*   **Multimodal:** `av`, `librosa`, `pillow`.
*   **Plotting:** `matplotlib`.

## 4. Core Modules (`src/llamafactory/`)

### 4.1. `hparams/` (Configuration)

*   Defines dataclasses (`ModelArguments`, `DataArguments`, `TrainingArguments`, `FinetuningArguments`, etc.) for structured configuration, leveraging `dataclasses` and `pydantic`.
*   `parser.py`: Central configuration hub using `transformers.HfArgumentParser`.
    *   Parses arguments from CLI, JSON, or YAML config files.
    *   Validates argument combinations across different dataclasses (e.g., checking stage vs. dataset type).
    *   Checks for optional dependencies based on selected features (e.g., `pip install .[torch]`).
    *   Applies necessary argument modifications or defaults based on other settings.
    *   Loads `dataset_info.json` to map dataset names to `DatasetAttr` metadata.
*   `DatasetAttr`: Dataclass holding metadata about each dataset (loading path, format, column names, ranking capability, etc.).

### 4.2. `model/` (Model Loading & Handling)

*   **Orchestration:** `loader.py` is the main entry point.
    *   `load_tokenizer`: Loads tokenizer (`AutoTokenizer`) and optionally processor (`AutoProcessor`), applies patches via `patcher.patch_tokenizer`.
    *   `load_config`: Loads model config (`AutoConfig`), applies patches via `patcher.patch_config`.
    *   `load_model`: Loads model weights (`AutoModelFor*`), handles quantization (via `quantization.py`), applies optimizations (Unsloth, Liger kernel), handles Mixture-of-Depths (MoD), calls `patcher.patch_model`, initializes PEFT adapters via `adapter.init_adapter`, and optionally adds/loads a value head (for RLHF stages) via `trl.AutoModelForCausalLMWithValueHead` and `patch_valuehead_model`. Selects appropriate AutoClass (CausalLM, Seq2SeqLM, Vision2Seq, etc.).
*   **Adapters:** `adapter.py` manages PEFT setup.
    *   `init_adapter`: Central function coordinating the setup based on `finetuning_type` (`lora`, `freeze`, `full`).
    *   Uses `peft` library (`get_peft_model`, `LoraConfig`) for LoRA, DoRA, LoRA+, PiSSA configuration.
    *   Handles freezing specific layers or full parameter tuning.
*   **Patching:** `patcher.py` applies dynamic modifications.
    *   `patch_tokenizer`: Adds special tokens, adjusts max length.
    *   `patch_processor`: Configures multimodal processor settings (image/video aspects).
    *   `patch_config`: Modifies config *before* model loading (dtype, attention impl, RoPE, quantization, MoE, KV cache). Relies heavily on `model_utils/` for specific feature patching.
    *   `patch_model`: Modifies model *after* loading (generation config, resize embeddings, gradient checkpointing, value head prep).
*   **Utilities:** `model_utils/` contains specialized modules for specific features/patches (attention implementations, checkpointing, embedding resizing, KV cache management, LongLoRA, quantization methods, RoPE scaling, Unsloth integration, value head logic, visual model specifics, MoD conversion/loading, Liger kernels).

### 4.3. `data/` (Data Pipeline)

*   **Goal:** Provide a unified pipeline: Load raw data -> Align to standard format -> Apply template -> Tokenize -> Collate batches.
*   **Orchestration:** `loader.py` manages the overall flow via `get_dataset`.
    *   `get_dataset`: Top-level function. Checks cache, loads raw data (`_load_single_dataset`), merges datasets if needed (`merge_dataset`), splits train/eval (`split_dataset`), and preprocesses via mapping (`_get_preprocessed_dataset`).
    *   `_load_single_dataset`: Loads data from various sources (HF Hub, MS Hub, OM Hub, local scripts, files) using `datasets.load_dataset` or specific hub libraries. Calls `align_dataset`.
    *   `_get_dataset_processor`: Selects the appropriate `DatasetProcessor` class (e.g., `SupervisedDatasetProcessor`) based on the training `stage`.
    *   `_get_preprocessed_dataset`: Applies the `process_dataset` method of the selected processor using `dataset.map()`.
*   **Parsing Metadata:** `parser.py` defines `DatasetAttr` and `get_dataset_list` to read `dataset_info.json` and retrieve metadata for requested datasets.
*   **Format Alignment:** `converter.py` standardizes raw datasets (from various formats like Alpaca, ShareGPT) into an internal format with standard keys (`_prompt`, `_response`, `_system`, etc.) using `align_dataset`.
*   **Prompt/Chat Templating:** `template.py` defines prompt/chat templates (`Template` dataclass).
    *   Specifies formatting rules (`Formatter` objects) for different roles (user, assistant, system, tool).
    *   `get_template_and_fix_tokenizer`: Retrieves the correct template instance based on `data_args.template`, configures tokenizer special tokens, and potentially sets the Jinja chat template.
*   **Stage-Specific Tokenization:** `processor/` contains `DatasetProcessor` subclasses.
    *   `supervised.py` (`SupervisedDatasetProcessor`, `PackedSupervisedDatasetProcessor`): Handles SFT data tokenization. Applies the template, tokenizes conversations, masks prompt tokens in labels (setting them to `IGNORE_INDEX`). The `Packed` version concatenates multiple examples into sequences for efficiency.
    *   Other files (`pretrain.py`, `pairwise.py`, `unsupervised.py`, `feedback.py`) handle PT, RM/DPO, PPO generation, and KTO stages respectively, implementing their specific tokenization needs.
*   **Batch Collation:** `collator.py` defines custom `DataCollator` classes.
    *   `MultiModalDataCollatorForSeq2Seq`: Base class handling padding and multimodal data integration (images, videos, audios) using `mm_plugin`.
    *   `SFTDataCollatorWith4DAttentionMask`: Used for SFT (especially packed SFT), handles padding and optionally creates 4D attention masks required by certain model implementations (like standard PyTorch SDPA when not using FlashAttention-2 with packing).
    *   `PairwiseDataCollatorWithPadding`: Structures batches for RM/DPO, containing pairs of chosen and rejected examples.
    *   `KTODataCollatorWithPadding`: Structures batches for KTO, including KL reference sequences and desirability tags.
*   **Utilities:** `data_utils.py` (shared helpers), `mm_plugin.py` (multimodal specifics), `tool_utils.py` (tool use formatting).

### 4.4. `train/` (Training Workflows)

*   **Orchestration:** `tuner.py` is the central coordinator.
    *   `run_exp`: Entry point for training runs. Parses args, sets up Ray Train if needed, calls `_training_function`.
    *   `_training_function`: Parses arguments into specific dataclasses, sets up callbacks (logging, reporting, optional SwanLab/PiSSA), and dispatches to the stage-specific `run_*` function (e.g., `run_sft`, `run_dpo`) based on `finetuning_args.stage`. Handles process group cleanup.
    *   `export_model`: Handles model export logic (adapter merging, quantization, saving locally or to Hub, Ollama Modelfile generation).
*   **Shared Utilities:** `trainer_utils.py` provides common functions.
    *   `create_modelcard_and_push`: Generates model card content, pushes model/tokenizer to Hub.
    *   `create_ref_model`, `create_reward_model`: Helpers to load reference/reward models for RLHF stages.
    *   `create_custom_optimizer`, `create_custom_scheduler`: Integration points for custom/experimental optimizers (GaLore, LoRA+, BAdam) and schedulers.
    *   `get_batch_logps`: Calculates log probabilities, essential for preference tuning methods (DPO, PPO, KTO).
    *   Ray/SwanLab integration helpers.
*   **Callbacks:** `callbacks.py` defines custom `transformers.TrainerCallback` implementations.
    *   `FixValueHeadModelCallback`: Ensures value head weights are correctly saved during checkpointing.
    *   `SaveProcessorCallback`: Saves multimodal processor alongside tokenizer.
    *   `PissaConvertCallback`: Handles PiSSA adapter initialization and saving.
    *   `LogCallback`: Provides enhanced logging (loss, LR, timing), progress bar updates, Web UI integration, and training interruption signal handling.
    *   `ReporterCallback`: Collects system information (CUDA, libs) and run arguments for reporting.
*   **Stage-Specific Logic (`sft/`, `dpo/`, `rm/`, `ppo/`, `kto/`, `pt/`)**:
    *   Each directory typically contains:
        *   `workflow.py`: Defines the main `run_*` function for the stage (e.g., `sft/workflow.py` contains `run_sft`). It orchestrates loading the model(s), tokenizer, dataset, collator, metrics, and initializing the custom trainer for that stage.
        *   `trainer.py`: Defines a custom trainer class (e.g., `sft/trainer.py` contains `CustomSeq2SeqTrainer`) inheriting from a base Hugging Face or TRL trainer (`Trainer`, `Seq2SeqTrainer`, `DPOTrainer`, `PPOTrainer`). It overrides methods to implement stage-specific loss calculation, prediction steps, metric computation, or optimizer/scheduler creation.
        *   `metric.py` (optional): Defines stage-specific evaluation metrics (e.g., accuracy, ROUGE, BLEU).
    *   **Example (SFT):**
        *   `sft/workflow.py` (`run_sft`): Loads SFT dataset/collator, model, tokenizer, initializes `ComputeAccuracy` metric, creates `CustomSeq2SeqTrainer`, calls `trainer.train()`, `trainer.evaluate()`, `trainer.predict()`.
        *   `sft/trainer.py` (`CustomSeq2SeqTrainer`): Inherits `Seq2SeqTrainer`, overrides prediction step for prompt masking, adds `save_predictions`.
    *   **Example (DPO):**
        *   `dpo/workflow.py` (`run_dpo`): Loads pairwise dataset (`stage="rm"`), policy and reference models (`create_ref_model`), initializes `CustomDPOTrainer`.
        *   `dpo/trainer.py` (`CustomDPOTrainer`): Inherits `trl.DPOTrainer`, implements DPO loss variants (Sigmoid, IPO, ORPO, SimPO), handles reference model forward pass, calculates rewards/metrics based on policy/ref logprobs.

## 5. How to Run Code

*   **Primary Interface:** `llamafactory-cli <command> [arguments]`. (Alternatively `python src/run.py ...` or `python src/llamafactory/cli.py ...`).
*   **Cluster Submission:** Use `python run.py config.yaml [cluster_args]` (e.g., `--ngpus 4 --walltime 24:00:00`). This script reads `config.yaml`, generates a run name and output directory, creates a PBS script using `tcsh`, and submits it via `qsub`. The submitted job then runs `llamafactory-cli train <generated_config_in_output_dir>`.
*   **Direct Execution:** Run `llamafactory-cli train config.yaml` for direct execution (e.g., on a single node or using `torchrun` manually). `cli.py` handles `torchrun` invocation automatically for multi-GPU non-Ray setups.
*   **Commands:** `train`, `eval`, `chat`, `webchat`, `webui`, `api`, `export`, `version`, `env`, `help`.
*   **Configuration:**
    *   Pass arguments via CLI (`--arg value`).
    *   **Recommended:** Use a single YAML/JSON config file (`llamafactory-cli train config.yaml`).
*   **Key Arguments:** `--stage` (e.g., `sft`, `dpo`), `--model_name_or_path`, `--dataset`, `--template`, `--finetuning_type` (e.g., `lora`, `full`), `--output_dir`, standard `TrainingArguments` (like `--learning_rate`, `--num_train_epochs`, `--per_device_train_batch_size`, etc.).
*   **Environment:** Install dependencies from `requirements.txt` (consider extras like `.[torch]`, `.[metrics]`).
*   **Distributed:** `torchrun` is typically handled automatically by `cli.py` for multi-GPU runs when not using Ray. `run.py` configures PBS to provide the necessary GPUs.

## 6. How to Reproduce Results

*   **Use Config Files:** Save all run settings in a YAML/JSON file and commit it.
*   **Set Seed:** Use `--seed <value>` for reproducibility across runs.
*   **Standardize Environment:** Use consistent dependency versions (ideally via Docker or a `requirements.lock` file). Note versions in logs.
*   **Hardware/Software:** Be aware of potential variations due to GPU type, CUDA version, library versions (especially `torch`, `transformers`, `flash-attn`). Record these in run notes.
*   **Check Artifacts:** Compare logs (`trainer.log`), metrics (`all_results.json`, WandB/MLflow runs), and generated checkpoints in the `--output_dir`.

## 7. How to Modify/Extend

*   **New Model:** Add template definition in `data/template.py`, potentially add model-specific patches in `model/patcher.py` or `model/model_utils/`. Ensure the AutoClass is correctly identified in `model/loader.py`.
*   **New Dataset:** Add entry to `data/dataset_info.json`. If the format is non-standard, implement a new `DatasetConverter` in `data/converter.py` and add a corresponding `formatting` option.
*   **New Training Stage:** Create `train/<stage>/` directory with `workflow.py` (defining `run_<stage>`) and `trainer.py` (defining `Custom<Stage>Trainer`). Add dispatch logic in `train/tuner.py::_training_function`. Potentially add a new `DatasetProcessor` in `data/processor/` and a `DataCollator` in `data/collator.py`. Update `hparams/finetuning_args.py` with the new stage option.
*   **New PEFT Method:** Modify `model/adapter.py` to integrate the method (likely using `peft` library). Update `hparams/finetuning_args.py` with the new option.
*   **UI/API:** Modify code within `src/llamafactory/webui/` or `src/llamafactory/api/`.

## 8. SFT Data Flow Summary (Detailed)

1.  **Load:** `loader._load_single_dataset` loads raw data based on `dataset_info.json` attributes.
2.  **Align:** `converter.align_dataset` standardizes column names/formats (e.g., to `_prompt`, `_response`).
3.  **Template Application & Tokenization:** `processor.SupervisedDatasetProcessor.process_dataset` (called via `dataset.map`):
    *   Retrieves template object (`template.Template`) via `get_template_and_fix_tokenizer`.
    *   Uses template's `encode_multiturn` (or similar) method to format the conversation string.
    *   Tokenizes the formatted string using `tokenizer`.
    *   Calculates labels, masking out prompt/input sections by setting label IDs to `IGNORE_INDEX`.
    *   Handles truncation and packing (`PackedSupervisedDatasetProcessor`) if enabled.
4.  **Collation:** `collator.SFTDataCollatorWith4DAttentionMask.__call__`:
    *   Receives a list of processed examples (dictionaries).
    *   Handles multimodal feature processing (if any) via `template.mm_plugin`.
    *   Pads `input_ids`, `attention_mask`, and `labels` to the maximum length in the batch using `tokenizer.pad`.
    *   If packing/block-diagonal attention is enabled (and not using FlashAttention-2), converts the 2D `attention_mask` to a 4D causal mask suitable for standard attention implementations using `prepare_4d_attention_mask`.
5.  **Training Step:** The batched data (dictionary of tensors) is passed to `sft.trainer.CustomSeq2SeqTrainer.compute_loss` (or the underlying model's forward method).
