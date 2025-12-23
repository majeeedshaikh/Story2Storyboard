# Story2Storyboard: Consistent Visual Storyboard Generation from Text Narratives

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-under%20review-yellow.svg)

**Automatically generate visually consistent storyboards from text narratives using IP-Adapter and SDXL-Turbo**

[Paper](#) • [Quick Start](#-quick-start) • [Dataset](#-dataset) • [Reproducibility](#-reproducibility) • [Citation](#-citation)

</div>

---

## 📋 Overview

**Story2Storyboard** is a novel approach for generating consistent visual storyboards from text narratives. The system addresses the critical challenge of maintaining visual consistency (character appearance, style, and setting) across multiple frames while accurately following the narrative progression.

### Key Contributions

- 🎯 **Visual Consistency**: Achieves **+15.3% improvement** in visual consistency (CLIP-I) compared to baseline text-only generation
- ⚡ **Efficient Generation**: Leverages SDXL-Turbo for fast inference (~30x faster than standard SDXL)
- 🔄 **Training-Free**: No model fine-tuning required - uses pre-trained SDXL-Turbo with IP-Adapter out-of-the-box
- 💰 **Cost-Effective**: Works without expensive training infrastructure
- 🔄 **Memory Injection**: Uses IP-Adapter to inject visual memory from reference frames, ensuring character and style consistency
- 📊 **Comprehensive Evaluation**: Validated on VinaBench (VWP subset) with quantitative metrics

---

## 🎬 Problem Statement

Traditional text-to-image generation models struggle with maintaining visual consistency across sequential frames. When generating storyboards from narratives, each frame is typically generated independently, leading to:

- **Character inconsistency**: Same characters appear differently across frames
- **Style drift**: Visual style changes between frames
- **Context loss**: No memory of previous frames

<div align="center">
  <img src="assets/problem_visulized.png" alt="Problem Visualization" width="800"/>
  <p><em>Visualization of the consistency problem in storyboard generation</em></p>
</div>

---

## 🏗️ System Architecture

Our approach uses **IP-Adapter** to inject visual memory from the first generated frame into subsequent frames, ensuring consistency while maintaining narrative accuracy.

<div align="center">
  <img src="assets/system_arch1.png" alt="System Architecture" width="900"/>
  <p><em>System architecture: IP-Adapter enables visual memory injection for consistent storyboard generation</em></p>
</div>

### Methodology: Auto-Regressive Visual Prompting

Our approach uses an **auto-regressive visual prompting** strategy:

1. **Anchor Generation** (Frame 0):
   - Generate first frame using text-only prompt (baseline SDXL-Turbo)
   - This serves as the "visual anchor" for consistency

2. **Visual Memory Injection** (Frames 1+):
   - Load IP-Adapter to enable visual conditioning
   - Inject the anchor frame as visual prompt for each subsequent frame
   - Balance between narrative adherence and visual consistency via adapter scale (λ = 0.5)

3. **Key Innovation**:
   - **Training-free**: No fine-tuning required - uses pre-trained models
   - **Identity preservation**: Maintains character appearance across frames
   - **Style consistency**: Preserves visual style throughout sequence

---

## 📊 Results

### Quantitative Evaluation on VinaBench (VWP Subset)

We evaluate our method on the VWP subset of VinaBench dataset using two key metrics: Visual Consistency (CLIP-I) and Text Alignment (CLIP-T). Evaluation was performed on 10 randomly sampled stories from the test set (834 total scenarios available).

| Method | Visual Consistency (CLIP-I) ↑ | Text Alignment (CLIP-T) ↑ |
|--------|-------------------------------|---------------------------|
| **Baseline (SDXL)** | 0.6683 | 0.2317 |
| **Ours (Auto-Regressive)** | **0.7706** | **0.2414** |
| **Improvement** | **+15.3%** | **+4.2%** |

### Key Findings

- ✅ **Significant improvement** in visual consistency (+15.3%)
- ✅ **Improved text alignment** (+4.2%) while maintaining consistency
- ✅ **Efficient inference** with SDXL-Turbo (1-2 steps vs 30+ steps)

---

## 🖼️ Examples

### Example 1: Character Consistency

<div align="center">
  <img src="assets/example1.png" alt="Example 1" width="900"/>
  <p><em>Maintaining character appearance across multiple frames</em></p>
</div>

### Example 2: Narrative Progression

<div align="center">
  <img src="assets/example2.png" alt="Example 2" width="900"/>
  <p><em>Following narrative progression while preserving visual style</em></p>
</div>

### Example 3: Complex Scenes

<div align="center">
  <img src="assets/example3.png" alt="Example 3" width="900"/>
  <p><em>Handling complex multi-character scenes with consistent styling</em></p>
</div>

---

## 🚀 Quick Start

1. **Open Notebook**: Launch `Story2Storyboard.ipynb` in Google Colab or Jupyter
2. **Install Dependencies**: Run the setup cells (automatically installs all packages)
3. **Load Data**: 
   - The repository includes `clean_test.json` and `test_images/` folder for reproducibility
   - No additional downloads needed - everything is ready to use!
4. **Run Generation**: Execute cells sequentially to generate storyboards
5. **Evaluate**: Run evaluation cells to compute CLIP-I and CLIP-T metrics

**Note**: For first-time setup, see [Installation](#-installation) section below.

---

## 💻 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (T4, V100, A100, or similar)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ free space for models and data

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+

### Inference Performance
- **Speed**: ~2-3 seconds per frame (SDXL-Turbo with 2 steps)
- **Batch Size**: 1 (sequential generation for consistency)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 20GB+ free disk space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/majeeedshaikh/Story2Storyboard.git
cd Story2Storyboard
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate peft
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
pip install pillow matplotlib numpy tqdm huggingface_hub
```

3. **Data is Ready!**
   - The repository includes `clean_test.json` and `test_images/` folder
   - No additional data downloads needed for evaluation
   - Everything is ready to use!

**Note**: The complete working implementation is provided in `Story2Storyboard.ipynb`. The notebook includes all necessary code for data preparation, model setup, generation, and evaluation.

---

## 💻 Usage

The complete implementation is provided in `Story2Storyboard.ipynb`. The notebook contains:

1. **Data Preparation**: Uses included `clean_test.json` and `test_images/` folder (no download needed!)
2. **Model Setup**: Initialize SDXL-Turbo and IP-Adapter
3. **Generation Pipeline**: 
   - Baseline method (text-only generation)
   - Our method (auto-regressive with IP-Adapter)
4. **Evaluation**: Calculate CLIP-I and CLIP-T metrics
5. **Visualization**: Compare results with ground truth

### Key Implementation Details

The notebook implements the auto-regressive approach:
- Generate the first frame using text-only prompt (baseline SDXL-Turbo)
- Load IP-Adapter weights
- Generate subsequent frames by injecting the reference frame as visual memory
- Each frame follows its corresponding narrative line while maintaining visual consistency

### Adjusting Consistency vs. Narrative Adherence

You can control the balance between visual consistency and narrative diversity by adjusting the IP-Adapter scale:

```python
pipe.set_ip_adapter_scale(0.5)  # Default: balanced
# Higher values (0.6-0.8) = more consistency, less narrative diversity
# Lower values (0.3-0.5) = more narrative diversity, less consistency
```

**Note**: All code in this repository is provided as-is from the research notebook. For the exact working implementation, please refer to `Story2Storyboard.ipynb`.

---

## 📁 Dataset

We evaluate on the **VWP (Visual Storytelling)** subset of the **VinaBench** dataset:

- **VWP Test Set**: 834 scenarios with 4,901 images
- **Evaluation**: 10 randomly sampled stories from the test set
- **Source**: Movie scenes with narrative text and corresponding storyboard frames

### Included Test Data

For reproducibility, this repository includes:
- **`clean_test.json`**: Curated test set with 834 stories (26MB)
- **`test_images/`**: Corresponding ground truth images organized by story ID (73MB)

**No additional downloads required!** The test data is ready to use for evaluation.

### Data Curation

The `clean_test.json` file contains a curated subset of the VWP test set:
- **Total Available**: 834 stories with 4,901 images
- **Curation Criteria**: 
  - All images must be present and accessible
  - Stories must have at least 3 frames for valid consistency testing
  - Synchronized with local image paths for reproducibility
- **Evaluation Subset**: 10 randomly sampled stories (as in paper)

Each story includes:
- Narrative text (script lines describing each frame)
- Corresponding ground truth images (actual movie frames)
- Character profiles for consistency validation

### Full Dataset

The full VinaBench dataset also includes:
- **StorySalon**: 1,678 stories with 23,008 images
- **VWP Train Set**: 12,486 stories from movies

**Dataset**: [VinaBench on Hugging Face](https://huggingface.co/datasets/Silin1590/VinaBench)

---

## 🔬 Evaluation

The evaluation process is implemented in `Story2Storyboard.ipynb` (Cells 22-24):

1. **Batch Generation** (Cell 22): 
   - Randomly samples 10 stories from the VWP test set
   - Generates storyboards using both baseline and our method
   - Saves results for metric calculation

2. **Metrics Calculation**:
   - **CLIP-I** (Cell 23): Visual consistency between consecutive frames using CLIP image embeddings
   - **CLIP-T** (Cell 24): Text-image alignment for each frame using CLIP text-image similarity

The evaluation results show our auto-regressive method achieves:
- **+15.3% improvement** in visual consistency (CLIP-I: 0.6683 → 0.7706)
- **+4.2% improvement** in text alignment (CLIP-T: 0.2317 → 0.2414)

---

## 🔬 Reproducibility

To reproduce the exact results from the paper:

1. **Use Included Test Set**: 
   - `clean_test.json`: Curated test set with 834 stories (included in repo)
   - `test_images/`: Corresponding ground truth images (included in repo)
   - No additional downloads needed!

2. **Set Random Seed**: 
   ```python
   import random
   import numpy as np
   import torch
   
   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)
   ```

3. **Run Evaluation**: 
   - Use `NUM_SAMPLES = 10` in Cell 22
   - This will randomly sample 10 stories (same as paper evaluation)

4. **Expected Results**:
   - Baseline CLIP-I: ~0.6683
   - Ours CLIP-I: ~0.7706
   - Improvement: ~+15.3%
   - Baseline CLIP-T: ~0.2317
   - Ours CLIP-T: ~0.2414
   - Improvement: ~+4.2%

---

## ⚠️ Limitations

- **Evaluation Scale**: Results reported on 10 sampled stories (not full 834 test set)
- **Domain**: Evaluated primarily on cinematic/movie narratives (VWP dataset)
- **Consistency Trade-off**: Higher adapter scale improves consistency but may reduce narrative diversity
- **Computational**: Requires GPU for reasonable inference time

---

## ❓ Frequently Asked Questions

**Q: Do I need to train the model?**  
A: No! This is training-free. We use pre-trained SDXL-Turbo with IP-Adapter.

**Q: Can I use my own stories?**  
A: Yes! Simply format your narrative as a list of strings and follow the notebook pipeline.

**Q: How do I adjust consistency vs. narrative adherence?**  
A: Modify `pipe.set_ip_adapter_scale()` - higher values (0.6-0.8) = more consistency, lower (0.3-0.5) = more narrative diversity.

**Q: Why only 10 stories in evaluation?**  
A: Computational constraints. The full test set (834 stories) is available in `clean_test.json` for larger-scale evaluation.

**Q: Do I need to download the dataset?**  
A: No! The test data (`clean_test.json` and `test_images/`) is included in this repository for reproducibility.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{majeed2025story2storyboard,
  title={Story2Storyboard: Consistent Visual Narrative Generation via Auto-Regressive Visual Prompting},
  author={Majeed, Muhammad Abdul and Haq, Abdul Wasay Ul},
  booktitle={2025 International Conference on IT and Industrial Technologies (ICIT)},
  year={2025},
  url={https://github.com/majeeedshaikh/Story2Storyboard}
}
```

**Note**: This paper is currently under review at a conference.

---

## 🛠️ Technical Details

### Models Used
- **Base Model**: [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) (Stability AI)
- **Adapter**: [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) (Tencent AI Lab)
- **Evaluation**: [CLIP](https://github.com/openai/CLIP) (OpenAI)

### Key Hyperparameters
- Inference steps: 2 (SDXL-Turbo)
- Guidance scale: 0.0 (SDXL-Turbo)
- IP-Adapter scale: 0.5
- Image resolution: 1024×1024

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) for SDXL-Turbo
- [Tencent AI Lab](https://github.com/tencent-ailab) for IP-Adapter
- [VinaBench Dataset](https://huggingface.co/datasets/Silin1590/VinaBench) creators
- The open-source community for excellent tools and libraries

---

## 📧 Contact

For questions or inquiries, please open an issue or contact:
- **GitHub**: [@majeeedshaikh](https://github.com/majeeedshaikh)
- **Repository**: [Story2Storyboard](https://github.com/majeeedshaikh/Story2Storyboard)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for the research community

</div>

