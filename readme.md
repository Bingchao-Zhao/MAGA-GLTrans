MAGA-GLTrans
===========
## Towards Computation-, Communication-efficient and Storage-friendly Computational Pathology.

Despite the remarkable performance across diverse applications, current computational pathology (CPath) models still suffer from the efficiency problem since they analyze whole-slide images at high magnification. It greatly hinders their clinical viability, particularly in clinical scenarios where there is a demand for fast diagnosis and efficient data transfer. Here we introduce a computation- and communication-efficient approach which we named **MAG**nification-**A**ligned **G**lobal-**L**ocal **Trans**former (MAGA-GLTrans), which can greatly reduces the computational time as well as the file transfer and storage overhead by allowing low magnification inputs instead of high magnification ones. To bridge the information gap between low and high magnification, magnification alignment (MAGA) is introduced to align the features from low magnification to high magnification in a self-supervised manner. By applying MAGA-GLTrans to various fundamental CPath tasks, we show that MAGA-GLTrans achieves state-of-the-art classification performance while achieving up to 10.7 times reduction in the computational time and over 20 times reduction in the file transfer and storage overhead. We further demonstrate the extensibility of MAGA to any CPath architecture improve their efficiency, and to histopathology-specific encoder to further improve the classification performance. A web-based platform (MediAIHub) is constructed to provide a feasible and efficient solution for telepathology for intraoperative frozen section diagnosis.

## Installation
- Linux (Tested on Ubuntu 22.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 3090)
- Python (3.10.15).
  
Detailed dependency requirements are described in requirements.txt.
``` shell
conda create -n gltrans python==3.10.15
conda activate gltrans
pip install -r requirements.txt
```
## Reconstruction WSIs
MAGA-GLTrans is trained and predicts at magnifications of 20X, 10X, and 5X. If WSIs do not have these magnifications, reconstruction is required. We use the `rebuild_multilevel_svs.py` script for WISs reconstruction. By default, the reconstructed images are saved in the `.svs` format. The CAMELYON 16 dataset has a complete pyramid structure, no reconstruction is required.
* `DATA_DIR`: The folder where the original WSI is located.
* `SAVE_DIR`: Reconfigure the save folder of the WSI.
* `SUFFIX`: The suffix of the original WSI file. such as‌ '.svs', '.tif', '.mrxs'.

### Tissue Segmentation {#ts}
The tissue segmentation process follows the [CLAM](https://github.com/mahmoodlab/CLAM).

### Feature Extraction
For ResNet50 feature extraction we used the same process as for [CLAM](https://github.com/mahmoodlab/CLAM).
For feature extraction of the UNI model, we refer to the official code of  [UNI](https://github.com/mahmoodlab/UNI).
For feature extraction of the CTransPath model, we refer to the official process of [CTransPath](https://github.com/Xiyue-Wang/TransPath).
It is worth noting that the corresponding WSI tile resolutions for this paper at 20X, 10X and 5X are 256×256, 128×128 and 64×64, respectively.

The features are saved as a dictionary in a `.npy` file. The specific format is: `data = {'feature':[f1,... ,fn], 'index':[[],... ,[]]}`.

### Magnification-Aligned
In the magnification alignment task, the input low-magnification image must maintain the same field of view as the high-magnification image. For example, in the 5X and 20X alignment tasks, if the image resolution of 20X is 256×256, the image resolution of 5X should be 64×64.

#### 1. Patching
We first use the `WSI_to_patches.py` script to split the WSI into Patches and record the file path of each Patch for subsequent data loading. Users can specify the WSI they want to slice by modifying the WSI storage path in the `config/align_conf.yaml` configuration script. In the `WSI_to_patches.py` script:
* `-ps`: Patch size.
* `-slide_ext`: The suffix of the WSI file. For example: `.svs`, `.tif`.
* `-X20_level`: Level of 20X image of the WSI file. Normally 0 or 1.

#### 2. Train Magnification Aligned Model
After patching of WSIs, we train the alignment model by the `mag_align_train.py` script. The alignment model needs to be loaded with pre-training weights for the high-magnification feature extractor. Pre-training weights for ResNet50 are available in [CLAM](https://github.com/mahmoodlab/CLAM). The weights are placed in the `weight` directory. In the code, we use the `ResNet50` model by default. We explain some important parameters. 
* `-dataset`: Dataset name.
* `-resolution`: The multiplicity of alignment is required. 
* `PATH_RECORD`: The `CSV` file that records the path to the wsi patches. 
* `SAVE_DIR`: The root of the WSI patches.

#### 3. Inference of Aligned Model.
* `-infer_wsi_dir`: WSI folder that requires inference.
* `-infer_wsi_index_dir`: Tissue index `CSV` file for the WSI. Requires the same file name as WSI. It can be obtained by `[Tissue Segmentation](#ts)` step.
* `-infer_weight`: The weight of align model.

### MAGA-GLTrans Training
**Before training or testing, the `data_path` parameter in the 'config/model_base_conf.yaml' script needs to be modified to specify the path of the features.**
The MAGA-Trans model is trained like the regular MIL model. Training with aligned features only requires replacing the original feature folder with the folder of aligned features. The MAGA-GLTrans model is trained with the `train.py` script.
* `-stage`: 'train' or 'infer'.
* `-resolu`: Magnification of the training feature. '20X', '10X', '10X-A', '5X', '5X-A'. '5X-A' is 5X aligned to 20X features and '10X-A' indicates 10X aligned to 20X features.
* `-dataset`: Dataset of the training feature.
* `-fold`: Fold of the ten-fold.

### MAGA_GLTrans Testing
The user only needs to modify the `-stage` parameter in the `train.py` script to `'infer'`.
