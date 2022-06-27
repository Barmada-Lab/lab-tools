#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LAB_TOOLS_BASE=$SCRIPT_DIR/..

git clone https://github.com/pyenv/pyenv $HOME/.pyenv

echo 'export PATH=$PATH:$HOME/.local/bin:$HOME/bin' >> $HOME/.bash_profile
echo 'export PATH=$HOME/.pyenv/shims:$PATH' >> $HOME/.bash_profile
echo 'export PATH=$HOME/.pyenv/bin:$PATH' >> $HOME/.bash_profile

echo 'export PATH=$HOME/.poetry/bin:$PATH' >> $HOME/.bash_profile

echo 'eval "$(pyenv init -)"' >> $HOME/.bash_profile

PATH=$PATH:$HOME/.local/bin:$HOME/bin
PATH=$HOME/.pyenv/shims:$PATH
PATH=$HOME/.pyenv/bin:$PATH

eval "$(pyenv init -)"

pyenv install 3.9.9 && pyenv global 3.9.9

curl -sSL https://install.python-poetry.org | python3 -

PATH=$HOME/.poetry/bin:$PATH

cd $LAB_TOOLS_BASE/notebooks && poetry install && poetry run ./nbkgen.sh
