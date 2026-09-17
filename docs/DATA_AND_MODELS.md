# Data and model access

No dataset images, masks derived as image files, model weights or feature tensors
are distributed in this package. Obtain data from their original releases and
check the applicable terms. Dataset filenames in manifests are necessary to
reproduce fixed folds/cohorts; they are not patient identifiers.

## Dataset sources and downloads

### Kather-5K / Colorectal Histology MNIST

- Original release: [Collection of textures in colorectal cancer histology](https://zenodo.org/records/53169), DOI [10.5281/zenodo.53169](https://doi.org/10.5281/zenodo.53169).
- Original study: Kather et al. (2016), [Multi-class Texture Analysis in Colorectal Cancer Histology](https://doi.org/10.1038/srep27988).
- Alternative distribution: [Kaggle, kmader/colorectal-histology-mnist](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist).
- Required original archive: `Kather_texture_2016_image_tiles_5000.zip` (5,000 RGB tiles, 150 x 150 pixels, eight tissue classes).
- Zenodo's archive MD5: `0ddbebfc56344752028fda72602aaade`.
- The separate `Kather_texture_2016_larger_images_10.zip` contains the larger images and is not the 5,000-tile classification input.

Required local layout:

```text
Colorectal Histology MNIST/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/<class>/<tile>
```

### CRC-VAL-HE-7K

- Original release: [Kather, Halama, and Marx (2018), Zenodo record 1214456](https://zenodo.org/records/1214456), DOI [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456).
- Alternative distribution: [Kaggle, imrankhan77/crc-val-he-7k](https://www.kaggle.com/datasets/imrankhan77/crc-val-he-7k).
- Required original archive: **`CRC-VAL-HE-7K.zip` only** (7,180 tiles, 224 x 224 pixels, nine source tissue classes).
- Zenodo's archive MD5: `2fd1651b4f94ebd818ebf90ad2b6ce06`.
- The same record also contains NCT-CRC-HE-100K archives. Those much larger training datasets are **not needed or used** in this study.

Required local layout:

```text
CRC-VAL-HE-7K/<source_class>/<tile>
```

### Provenance and compatibility

The original Zenodo records are the primary citation and provenance sources; the
Kaggle links are alternative access points supplied for convenience. Their archive
and tile contents have not been checked for byte identity against Zenodo. The MD5
values above apply only to the named original Zenodo ZIP files, not repackaged
Kaggle downloads. They are download-integrity checks, not a substitute for the
experiment's saved tile inventories, mappings, and hashes. Check class folders,
tile names, dimensions, and available hashes before reusing fixed splits or models.
Do not regenerate the study's cohorts silently to accommodate a different copy.

Credit the original dataset authors and publications and comply with the terms
on the source distribution pages. No dataset license is granted by this repository.

- The external seven-class evaluation uses 6,588 compatible tiles, not a native
  nine-class benchmark. Smooth muscle is excluded; DEB and MUC share a coarse
  debris/mucus target; Kather complex stroma is excluded from seven-class training.
- External models are refitted on seven-class Kather data before inference on CRC.
  They are not trained on CRC or NCT-CRC-HE-100K. No external patient cluster IDs
  are available in the distributed metadata used by this study.

## Model access

ResNet18 weights are obtained through torchvision, DINOv2 through the official
model source used by the loader, and UNI through the authorized MahmoodLab model
repository. UNI approval/authentication must be obtained separately. Do not put
tokens into notebook cells, commits, configuration files or this repository.

For UNI's timm/Hugging Face path, leave the local assets argument unset and use
authorized cached/downloaded weights. The alternate local-asset loader additionally
requires the upstream UNI Python package. Historical MaskCut reproduction requires
the upstream MaskCut/CutLER implementation expected by the wrapper. Those upstream
repositories and their licenses must be obtained separately, not copied from the
working project's vendored folders without review.
