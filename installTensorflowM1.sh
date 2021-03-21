cd ~/Downloads
wget https://github.com/conda-forge/miniforge/releases/download/4.9.2-7/Mambaforge-4.9.2-7-MacOSX-arm64.sh
chmod +x Mambaforge-4.9.2-7-MacOSX-arm64.sh
./Mambaforge-4.9.2-7-MacOSX-arm64.sh
which python
file $(which python)
wget https://raw.githubusercontent.com/mwidjaja1/DSOnMacARM/main/environment.yml
conda env create --file=/Users/hannes/Downloads/environment.yml --name=condaVenv
conda activate condaVenv
pip3 install --upgrade --force --no-dependencies \
  https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_$(uname -m).whl \
  https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_$(uname -m).whl
python -c 'import tensorflow as tf; print(tf.__version__)'
conda install opencv
python -c 'import tensorflow as tf; import cv2; print(cv2.__version__); print(tf.__version__)'