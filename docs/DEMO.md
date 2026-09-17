# Dataset-free explanation demonstration

The site in `docs/index.html` is a static educational demonstration. It runs directly
from a local file or from GitHub Pages, without a server, dependencies, accounts,
images, weights, analytics, or uploads. No data leave the browser. Repository links
navigate to GitHub; normal hosting request logs may still be collected by the host.

## Scope

The interactive experiment uses an explicit additive scoring function, not ResNet18,
DINOv2, or UNI. It cannot demonstrate their classification or attribution performance.
The workflow diagram describes the actual study; the interactive toy only explains
why perturbation tests are needed. No biological-causality claim is made.

`toy-model.js` defines the seeded synthetic input and scoring function. Positive
weights occupy a disk and negative weights a separate rectangle. Other weights are
zero. Occlusion replaces an input patch value with zero. Exact signed contributions
form the faithful map; an unrelated spatial bump forms the misleading map. Each
map's ranking remains fixed during deletion. Exact ties use ascending patch index.
Twenty seeded random orders are shared across maps. The random preview displays the
first draw; the random curve averages all twenty. CSV exports identify their content
as `synthetic_additive_toy` and include the seed, map, baseline, and all deletion steps.

## Test

With Node.js installed, from the repository root:

```bash
node scripts/test_toy_demo.cjs
```

The test covers signed single-patch interventions, ties, complete deletion, exact
score arithmetic, shared random controls, and the misleading-map counterexample.
The demo has no npm installation or compilation step.

## Publish on GitHub Pages

In repository **Settings > Pages**, choose **Deploy from a branch**, then **main**
and **/docs**, and save. The expected address is:

https://haocht10032.github.io/cancer_image_pathology/

The `.nojekyll` file keeps this a plain static site. No GitHub Actions workflow or
GPU hosting is required. The README diagram renders independently of Pages being
enabled. No restricted data or weights should ever be added to the demo directory.
