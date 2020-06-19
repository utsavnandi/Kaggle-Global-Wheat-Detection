pip uninstall kaggle -y
pip install kaggle -q
pip install -U albumentations -q
pip install -U git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer -q
pip install -U git+https://github.com/rwightman/efficientdet-pytorch -q
pip install -U git+https://github.com/rwightman/pytorch-image-models -q
pip install --upgrade omegaconf -q
pip install neptune-client -q
mkdir ~/.kaggle/
cp ./kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download global-wheat-detection
mkdir ./data/
mv global-wheat-detection.zip ./data/global-wheat-detection.zip
unzip ./data/global-wheat-detection.zip -d ./data/
rm ./data/global-wheat-detection.zip