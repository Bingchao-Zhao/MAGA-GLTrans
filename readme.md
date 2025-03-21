MAGA-GLTrans
===========
## Towards Computation-, Communication-efficient and Storage-friendly Computational Pathology.
Despite the impressive performance across a wide range of applications, current computational pathology (CPath) models face significant diagnostic efficiency challenges due to their reliance on high-magnification whole-slide image analysis. This limitation severely compromises their clinical utility, especially in time-sensitive diagnostic scenarios and situations requiring efficient data transfer. To address these issues, we present a novel computation- and communication-efficient framework called **MAG**nification-**A**ligned **G**lobal-**L**ocal **Trans**former (MAGA-GLTrans). Our approach significantly reduces computational time, file transfer requirements, and storage overhead by enabling effective analysis using low-magnification inputs rather than high-magnification ones. The key innovation lies in our proposed magnification alignment (MAGA) mechanism, which employs self-supervised learning to bridge the information gap between low and high magnification levels by effectively aligning their feature representations. Through extensive evaluation across various fundamental CPath tasks, MAGA-GLTrans demonstrates state-of-the-art classification performance while achieving remarkable efficiency gains: up to 10.7&times; reduction in computational time and over 20&times; reduction in file transfer and storage requirements. Furthermore, we highlight the versatility of our MAGA framework through two significant extensions: (1) its applicability as a feature extractor to enhance the efficiency of any CPath architecture, and (2) its compatibility with existing foundation models and histopathology-specific encoders, enabling them to process low-magnification inputs with minimal information loss. These advancements position MAGA-GLTrans as a particularly promising solution for telepathology applications, especially in the context of intraoperative frozen section diagnosis where both accuracy and efficiency are paramount.

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

## Tissue Segmentation 
The tissue segmentation process follows the [CLAM](https://github.com/mahmoodlab/CLAM).

## Feature Extraction
For ResNet50 feature extraction we used the same process as for [CLAM](https://github.com/mahmoodlab/CLAM).
For feature extraction of the UNI model, we refer to the official code of  [UNI](https://github.com/mahmoodlab/UNI).
For feature extraction of the CTransPath model, we refer to the official process of [CTransPath](https://github.com/Xiyue-Wang/TransPath).
It is worth noting that the corresponding WSI tile resolutions for this paper at 20X, 10X and 5X are 256×256, 128×128 and 64×64, respectively.

The features are saved as a dictionary in a `.npy` file. The specific format is: `data = {'feature':[f1,... ,fn], 'index':[[],... ,[]]}`.

## Magnification-Aligned
In the magnification alignment task, the input low-magnification image must maintain the same field of view as the high-magnification image. For example, in the 5X and 20X alignment tasks, if the image resolution of 20X is 256×256, the image resolution of 5X should be 64×64.

### 1. Patching
We first use the `WSI_to_patches.py` script to split the WSI into Patches and record the file path of each Patch for subsequent data loading. Users can specify the WSI they want to slice by modifying the WSI storage path in the `config/align_conf.yaml` configuration script. In the `WSI_to_patches.py` script:
* `-ps`: Patch size.
* `-slide_ext`: The suffix of the WSI file. For example: `.svs`, `.tif`.
* `-X20_level`: Level of 20X image of the WSI file. Normally 0 or 1.

### 2. Train Magnification Aligned Model
After patching of WSIs, we train the alignment model by the `mag_align_train.py` script. The alignment model needs to be loaded with pre-training weights for the high-magnification feature extractor. Pre-training weights for ResNet50 are available in [CLAM](https://github.com/mahmoodlab/CLAM). The weights are placed in the `weight` directory. In the code, we use the `ResNet50` model by default. We explain some important parameters. 
* `-dataset`: Dataset name.
* `-resolution`: The multiplicity of alignment is required. 
* `PATH_RECORD`: The `CSV` file that records the path to the wsi patches. 
* `SAVE_DIR`: The root of the WSI patches.

### 3. Inference of Aligned Model.
* `-infer_wsi_dir`: WSI folder that requires inference.
* `-infer_wsi_index_dir`: Tissue index `CSV` file for the WSI. Requires the same file name as WSI. It can be obtained by `Tissue Segmentation` step.
* `-infer_weight`: The weight of align model.

## MAGA-GLTrans Training
**Before training or testing, the `data_path` parameter in the 'config/model_base_conf.yaml' script needs to be modified to specify the path of the features.**
The MAGA-Trans model is trained like the regular MIL model. Training with aligned features only requires replacing the original feature folder with the folder of aligned features. The MAGA-GLTrans model is trained with the `train.py` script.
* `-stage`: 'train' or 'infer'.
* `-resolu`: Magnification of the training feature. '20X', '10X', '10X-A', '5X', '5X-A'. '5X-A' is 5X aligned to 20X features and '10X-A' indicates 10X aligned to 20X features.
* `-dataset`: Dataset of the training feature.
* `-fold`: Fold of the ten-fold.

## MAGA_GLTrans Testing
The user only needs to modify the `-stage` parameter in the `train.py` script to `'infer'`.
