# Data and model access

No dataset images, masks derived as image files, model weights or feature tensors
are distributed in this package. Obtain data from their original releases and
check the applicable terms. Dataset filenames in manifests are necessary to
reproduce fixed folds/cohorts; they are not patient identifiers.

- Kather-5K: Kather et al. (2016), DOI `10.1038/srep27988`. Required local layout:
  `Colorectal Histology MNIST/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/<class>/<tile>`.
- CRC-VAL-HE-7K: Zenodo DOI `10.5281/zenodo.1214456`. Required layout:
  `CRC-VAL-HE-7K/<source_class>/<tile>`.
- The external seven-class evaluation uses 6,588 compatible tiles, not a native
  nine-class benchmark. Smooth muscle is excluded; DEB and MUC share a coarse
  debris/mucus target; Kather complex stroma is excluded from seven-class training.
- External models are refitted on seven-class Kather data before inference on CRC.
  They are not trained on CRC or NCT-CRC-HE-100K. No external patient cluster IDs
  are available in the distributed metadata used by this study.

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
