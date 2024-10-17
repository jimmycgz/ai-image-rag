brew install miniconda

python3 --version

echo 'eval "$(/opt/homebrew/Caskroom/miniconda/base/bin/conda shell.zsh hook)"' >> ~/.zshrc
source ~/.zshrc

conda create --prefix ./env python=3.12.6 -y

conda init

conda activate ./env

