# Third-party notices and scope exceptions

Checked public upstream sources on 2026-10-02. Third-party materials are not
covered by the project's MIT or CC BY 4.0 grants. Branch URLs are references,
not a substitute for verifying the specific versions actually used in the study.
This notice records known exceptions and remaining provenance checks; it is not
a certification that every upstream requirement has been cleared.

| Component | Public upstream terms observed | Project handling |
| --- | --- | --- |
| CutLER / MaskCut | CC BY-NC-SA 4.0 in the CutLER repository LICENSE | Upstream tree is not bundled, but review the compact project implementation for adapted expression before claiming MIT |
| DINOv2 | Apache-2.0 in repository LICENSE | Encoder obtained separately; retain applicable upstream notices for any copied/adapted code and verify weight terms separately |
| UNI | CC BY-NC-ND 4.0 in repository LICENSE | Upstream tree and weights are not bundled; follow separately accepted model-access terms and review rights for any copied code or derived restricted outputs |
| Histology datasets | Original dataset release terms | Images are obtained separately; no project license is granted for them |
| Installed Python/R libraries | Each dependency's own license | Not relicensed by the project; audit vendored/adapted parts if found |

Official sources:

- https://github.com/facebookresearch/CutLER/blob/main/LICENSE
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE
- https://github.com/mahmoodlab/UNI/blob/main/LICENSE
- https://zenodo.org/records/53169
- https://zenodo.org/records/1214456

## Specific review items

1. `Methods/BaselineCNN/maskcut_preprocess.py` describes its compact graph
   partition as matching the upstream core flow. Determine whether its expression
   was independently implemented from the mathematical method or adapted from
   upstream source. Algorithmic similarity alone does not decide copyright
   provenance. Do not label it MIT by default. Preserve required upstream terms,
   obtain permission, or revise the future publication scope as appropriate.
2. Review `Methods/UNIAttribution/attribution.py`, encoder loaders, and other
   helpers for copied code and notices. A paper citation or implementation of a
   published equation is not itself proof that source code was copied or cleared.
3. Confirm what the separately accepted UNI access/model terms permit for
   checkpoints, embeddings, and derived audit outputs. Repository-license names
   alone are not a complete analysis of those terms. Do not assume all numeric
   outputs are restricted, or that all are freely redistributable.
4. Preserve existing notices and versions for any adapted PyTorch, torchvision,
   timm, attribution-library, or other dependency code. Ordinary imports do not
   make the dependency the project's property.
5. Keep original datasets, model weights, private correspondence, and raw executed
   notebook images out of any numerical archive unless separately cleared.

Research authorship does not establish the copyright holder for every component.
Consult the relevant institution/rights holder where ownership is uncertain.
