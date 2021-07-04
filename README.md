# This project is forked by [This project](https://github.com/v-iashin/BMT)

## Dense Video Captioning with Bi-modal Transformer

## Getting Started

This project was developed by adding training data from the original project and giving exception conditions. If you want to use the original project, go to the [link](https://github.com/v-iashin/BMT)

_The code is tested on `Ubuntu 16.04/18.04` with one `NVIDIA GPU 1080Ti/2080Ti`. If you are planning to use it with other software/hardware, you might need to adapt `conda` environment files or even the code._

Clone the repository. Mind the `--recursive` flag to make sure `submodules` are also cloned (evaluation scripts for Python 3 and scripts for feature extraction).
```bash
git clone --recursive https://github.com/yunsujeon/BMT.git
```

Download features (I3D and VGGish) and word embeddings (GloVe). The script will download them (~10 GB) and unpack into `./data` and `./.vector_cache` folders. *Make sure to run it while being in BMT folder*
```bash
bash ./download_data.sh

```
Set up a `conda` environment
```bash
conda env create -f ./conda_env.yml
conda activate bmt
# install spacy language model. Make sure you activated the conda environment
python -m spacy download en
```

## Add Train Data
Additionally, we created an additional dataset fitted to the movie scene. If you want, You can download the feature extracted npy file from the link 
[i3d_features](https://drive.google.com/drive/folders/1UTVkkgowg5wriGvX5vEuREAmBtnsg17L?usp=sharing)
[vggish_features](https://drive.google.com/drive/folders/1xTI488LitnyNrBwL8PTWlBx0-id_m_0A?usp=sharing)

If you want to use this extra data for training, you'll need to do some extra work.
1. Copy the files in the i3d folder to /BMT/data/i3d_25fps_stack64step64_2stream_npy/
2. Copy the files in the vggish folder to /BMT/data/vggish_npy/

You can add more train data to run feature extraction.


## Train

We train our model in two staged: training of the captioning module on ground truth proposals and training of the proposal generator using the pre-trained encoder from the captioning module.

We changed two parameters to reduce timespan.
1. --feature_timespan_in_fps (64->36)
2. --fps_at_extraction (25->15)

```bash
python main.py \
    --procedure train_cap \
    --B 32
```

- *Train proposal generation module*.
```bash
python main.py \
    --procedure train_prop \
    --pretrained_cap_model_path /your_exp_path/best_cap_model.pt \
    --B 16
```

## Single Video Prediction

*Disclaimer: we do not guarantee perfect results nor recommend you to use it in production. Sometimes captions are redundant, unnatural, and rediculous. Use it at your own risk.*

Extract I3D features
```bash
# run this from the video_features folder:
cd ./submodules/video_features
conda deactivate
conda activate i3d
python main.py \
    --feature_type i3d \
    --on_extraction save_numpy \
    --device_ids 0 \
    --extraction_fps 25 \
    --video_paths ../../sample/women_long_jump.mp4 \
    --output_path ../../sample/
```

Extract VGGish features (if `ValueError`, download the vggish model first--see `README.md` in `./submodules/video_features`)
```bash
conda deactivate
conda activate vggish
python main.py \
    --feature_type vggish \
    --on_extraction save_numpy \
    --device_ids 0 \
    --video_paths ../../sample/women_long_jump.mp4 \
    --output_path ../../sample/
```

Run the inference
```bash
# run this from the BMT main folder:
cd ../../
conda deactivate
conda activate bmt
python ./sample/single_video_prediction.py \
    --prop_generator_model_path ./sample/best_prop_model.pt \
    --pretrained_cap_model_path ./sample/best_cap_model.pt \
    --vggish_features_path ./sample/women_long_jump_vggish.npy \
    --rgb_features_path ./sample/women_long_jump_rgb.npy \
    --flow_features_path ./sample/women_long_jump_flow.npy \
    --duration_in_secs 35.155 \
    --device_id 0 \
    --max_prop_per_vid 100 \
    --nms_tiou_thresh 0.4
```


## Citation
Our paper was accepted at BMVC 2020. Please, use this bibtex if you would like to cite our work
```
@InProceedings{BMT_Iashin_2020,
  title={A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer},
  author={Iashin, Vladimir and Rahtu, Esa},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2020}
}
```

```
@InProceedings{MDVC_Iashin_2020,
  title = {Multi-Modal Dense Video Captioning},
  author = {Iashin, Vladimir and Rahtu, Esa},
