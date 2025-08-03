# Local Logit Linearity Analysis

> **Module:** `src/llamafactory/analysis/logit_linearity.py`  
> **CLI command:** `llamafactory-cli analyze-logits <config.yaml>`

---

## 1  Motivation
Neural‐network updates are often locally *linear* in small neighbourhoods of parameter space.  Quantifying this property helps us

* understand optimisation stability after fine-tuning,
* decide safe learning-rate ranges,
* validate theoretical assumptions (e.g. NTK regime), and
* debug anomalous checkpoints whose logit map turns highly non-linear.

The new tool measures the *largest* parameter‐space radius at which the model’s last-token logit function remains approximately first-order linear.

---

## 2  Mathematical formulation
Let \(\theta\in\mathbb R^{d}\) be the model parameters and \(f\colon\mathbb R^{d}\to\mathbb R^{V}\) map them to the *last-token logits* for a fixed input prompt \(x\).  Denote the Jacobian at the fine-tuned checkpoint \(\theta_{*}\) by
\[
J\_{*}(x)=\left.\frac{\partial f(x;\theta)}{\partial\theta}\right\vert\_{\theta=\theta_{*}}\;\in\;\mathbb R^{V\times d}.
\]
For a perturbation direction \(\delta\in\mathbb R^{d}\) of unit norm and *radius* \(r>0\) we compare

* **True** logit change  \(\;\Delta\_\text{true}=f(x;\theta_{*}+r\,\delta)-f(x;\theta_{*})\),
* **First-order prediction**  \(\;\Delta\_\text{pred}=r\,J\_{*}(x)\,\delta\;=\;\text{JVP}\bigl(f,\theta_{*};r\delta\bigr)\).

We aggregate over random directions \(\delta\) and prompts \(x\) the following metrics:

1. **Relative error**  \(\displaystyle \varepsilon=\frac{\lVert \Delta\_\text{true}-\Delta\_\text{pred}\rVert}{\lVert\Delta\_\text{true}\rVert}\;\in[0,\infty)\).
2. **Cosine similarity**  \(c=\cos\bigl(\Delta\_\text{true},\,\Delta\_\text{pred}\bigr) \in[-1,1].\)

Define user thresholds \(\varepsilon\_{\max}\) and \(c\_{\min}\).  The *linearity radius* \(r\_{\text{lin}}\) is the largest tested radius satisfying
\[
\mathbb E\bigl[\varepsilon\bigr]\le \varepsilon\_{\max}\quad\text{and}\quad\mathbb E\bigl[c\bigr]\ge c\_{\min}.
\]

---

## 3  Algorithmic workflow

1. **Load checkpoint** — full-parameter fine-tuned model only (no LoRA / adapters).
2. **Select parameters** — either the entire network or the last *k* blocks via `param_scope`.
3. **Sample directions**
   * Draw \(\delta\sim\mathcal N(0,\sigma^{2}I)\) restricted to the chosen parameters.
   * Globally normalise to unit Euclidean norm.
4. **Prepare probe inputs**
   * Load `datasets` slice or fallback hard-coded prompts.
   * Tokenise up to `cutoff_len`.
5. **For every radius** \(r\in\texttt{radii}\)
   * For each direction and batch of prompts
     1. Compute \(\Delta\_\text{true}\) via a *paired* forward pass with `functional_call`, adding and then removing the parameter delta in-place to avoid cloning the model.
     2. Compute \(\Delta\_\text{pred}\) with `torch.func.jvp`, which yields the Jacobian–vector product efficiently without materialising \(J\_{*}\).
     3. Store relative error & cosine similarity.
6. **Aggregate statistics** — mean ± std over all (direction, prompt) pairs.
7. **Determine linearity radius** according to user thresholds.
8. **Write JSON report** (config, per-radius metrics, inferred \(r\_{\text{lin}}\)).

Computations are done in fp32 for the JVP path when necessary to preserve numerical stability.

---

## 4  Usage
```bash
# Example (see provided YAML file)
llamafactory-cli analyze-logits configs/analysis/llama_math_linearity.yaml
```

The YAML file follows standard SFT configs; analysis-specific keys may reside at the root or under an `analysis:` section.

### Important arguments
| Key | Description | Default |
|---|---|---|
| `perturbation_std` | Std-dev of initial Gaussian directions | `1e-3` |
| `num_directions` | How many random directions to probe | `8` |
| `radii` | List of radii to test | `[0.1,0.3,1.0]` |
| `param_scope` | `full` or `last_blocks:k` | `full` |
| `cos_threshold` | Minimum mean cosine similarity | `0.95` |
| `relerr_threshold` | Maximum mean relative error | `0.20` |

### Output
`<output_dir>/<report_filename>` contains for each radius:
```json
{
  "0.3": {
    "mean_relative_error": 0.12,
    "std_relative_error": 0.04,
    "mean_cosine_similarity": 0.97,
    "std_cosine_similarity": 0.01
  },
  ...
}
```
`linearity_radius` gives the largest radius meeting both thresholds.

---

## 5  Implementation highlights
* **Minimal dependencies** — only `torch`, `transformers`, `datasets`, `omegaconf`.
* **`torch.func` API** — keeps the model static in VRAM and avoids expensive backward passes.
* **Batching** — prompts are processed jointly; directions are iterated but could be batched further for GPUs with spare memory.
* **Memory-safe deltas** — in-place `tensor.add_` avoids extra parameter copies; rollback after each perturbation ensures reuse.
* **Config-agnostic** — leverages the existing YAML parser pattern but ignores unknown keys, so future additions will not break the script.

---

## 6  Extending the analysis
* **Other layers** — modify `_select_param_indices` to probe embeddings only, attention matrices, etc.
* **Alternative metrics** — plug additional similarity measures in `_aggregate_stats`.
* **Second-order tests** — `torch.func.hvp` can be integrated to measure curvature effects.
* **Visualisation** — pair with a notebook to plot metric curves over radii.

---

## 7  File/command overview
| File | Purpose |
|------|---------|
| `src/llamafactory/analysis/logit_linearity.py` | Core analysis implementation |
| `configs/analysis/llama_math_linearity.yaml` | Example config |
| `src/llamafactory/cli.py` | Registers `analyze-logits` sub-command |

---

## 8  Changelog (late-ntk-analysis branch)
1. **Feature**: Add `analysis/logit_linearity.py`.
2. **CLI**: Register `analyze-logits` command.
3. **Config**: Provide example YAML under `configs/analysis/`.
4. **Docs**: Current document added at `docs/logit_linearity_analysis.md`.

---

© 2025 LLaMA-Factory team – released under Apache 2.0