# Creating group-level models of perceived attributes

This repository is a worked example of creating a group-level generative model of a perceived attribute, to characterize what a given (sub-)population thinks that attribute looks like. Given a set of stimuli and their averaged ratings, the notebook fits a linear model and uses it to edit a GAN so it can synthesize new stimuli with the modeled attribute. The generator is the official StyleGAN2 (SG2) model trained on the FFHQ dataset.

The ratings used here are for demonstration purposes and do not come from human judgments. We generated 1000 random faces using SG2 and had the InsightFace model predict their ages in years. The logic is the same as for human judgments, so you can also use this approach to interrogate computational models' representations.

All the methods here are covered in our 2022 PNAS paper:

Peterson, J. C., Uddenberg, S., Griffiths, T. L., Todorov, A., & Suchow, J. W. (2022). Deep models of superficial face judgments. _Proceedings of the National Academy of Sciences_, _119_(17), e2115228119. [PDF](https://www.socialperception.org/pdf/Peterson-et-al-2022-PNAS-Faces.pdf)

In principle, you could model judgments at the individual level with this technique, but it would be very noisy. As you might expect, the more individual image ratings you have, the better.

## What is in this repo

- `attribute_model_demo.ipynb` loads the ratings, fits the model, predicts a new face, and edits a face along the fitted direction.
- `demo_attribute_means.csv` holds the demo ratings.
- `demo_latents.npz` holds the StyleGAN2 latent for each stimulus.
- `Dockerfile` builds a GPU notebook image with StyleGAN2, the FFHQ weights, and InsightFace.

The StyleGAN2 code and FFHQ weights are downloaded when the image is built. They are not stored in this repo.

## Getting started

You will need an NVIDIA GPU and [Docker](https://docs.docker.com/engine/install/) with GPU support.

From this directory, build and run the container:

```
docker build -t attribute-demo .
docker run --gpus all -p 8888:8888 -v "${PWD}:/work" attribute-demo
```

The first command builds the image and names it `attribute-demo`. This will download all the necessary pieces to get the repository working (including StyleGAN2 and InsightFace dependencies), so it may take some time. The second command runs the container (mounting your current directory as `/work`) and mkaes the notebook available at http://127.0.0.1:8888.

(After the `docker run` command, your terminal may print some other URL, such as `http://e1dce6f67787:8888/` but I was never able to get the anonymous URL to work). Simply go to http://127.0.0.1:8888 in your browser, then click on `attribute_model_demo.ipynb` to open the notebook.

If the StyleGAN weights are missing for some reason, the notebook prints this download URL and stops: https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl

## Demo data

The notebook reads `demo_attribute_means.csv` and `demo_latents.npz` for the purposes of this demo. However, the code should work with your own data and latents as well, once formatted in the same way as those examples. Simply point the `ratings_csv` and `latents_npz` attributes in the `Config` defined in the notebook at your particular files when you want to use your own stimuli and ratings.

The ratings data should have a `stimulus` column (the stimulus index) plus one column per attribute. Values in an attribute column are the average rating from several raters. This demo has one attribute: `age` (the age output of InsightFace's `genderage` model)

`demo_latents.npz` is a numpy file with two arrays: `stimulus` and `latent`. `stimulus` is the stimulus index, matching that of the `stimulus` column in the ratings data. Each cell in the `latent` array is one 512-D vector from SG2 (so it represents a single face). However, SG2 expects an 18 x 512 array of weights, so the same 512-D vector is repeated 18 times. We join the ratings and the latents on the `stimulus` index.

You can use the `write_demo_data()` function to generate the demo data from scratch, for the sake of reproducibility. Given that the data are included in this repo, you can skip the call to `write_demo_data()` in the notebook.

For a real study with human judgments, plan on collecting approximately 30 unique ratings per face for a stable average. Each participant can comfortably judge about 100 images in a single online session with a slider (potentially many more images for an in-lab study). The ratings spreadsheet used by the notebook should contain the averaged rating for each stimulus across participants.

## Notebook overview

After finishing setup, this notebook will:

1. Load the stimuli and their ratings from the included files.
2. Fit a ridge regression model for each attribute. Ratings are converted from their native scale to z-scores (mean of 0, SD of 1) first. The penalty term is chosen by leave-one-out search over a log grid, and we report the leave-one-out R^2 as the metric of model fit. The fitted `direction` is the coefficient vector scaled so that adding it to a latent raises the predicted z-score by 1 (conversely, subtracting it lowers the z-score by 1).
3. Predict the rating of a new face that was not in the original training sample. This prediction is provided both in terms of its z-score and the original scale (in this case, years of age).
4. Build a results reel of the same new identity at different levels of the modeled attribute (i.e., age). Each frame of the reel is accompanied by its predicted z-score and corresponding rating. The exact predicted numbers can be inaccurate, but the face should still visibly change along the modeled attribute (going from younger to older as we move from left to right in the reel). Note that transformation tends to work better than prediction.

## Licenses

This repository is released under the MIT License. See [LICENSE](LICENSE).

NVIDIA's StyleGAN2 license is [here](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).

The InsightFace library code is MIT-licensed, but the pretrained `buffalo_l` weights are available for non-commercial research only. See [here](https://github.com/deepinsight/insightface/blob/v0.7/model_zoo/README.md) for details.

## Contact

If you would like to learn more about this work or use some of our existing models, please reach out to Stefan Uddenberg at stefanu@illinois.edu.
