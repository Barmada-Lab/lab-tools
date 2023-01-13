#!/bin/bash
#
set -e

TURBO=/nfs/turbo/umms-sbarmada

if [ ! -e $HOME/turbo ]; then
    ln -s $TURBO $HOME/turbo
fi    

if [ ! -e $HOME/Desktop/turbo ]; then
    ln -s $TURBO $HOME/Desktop/turbo
fi    

if [ ! -e $HOME/Desktop/ImageJ2.desktop ]; then
    cp $TURBO/shared/Fiji.app/ImageJ2.desktop $HOME/Desktop
fi

cp $TURBO/shared/.bash_profile $HOME

rm $HOME/.pyenv -rf
git clone https://github.com/pyenv/pyenv $HOME/.pyenv

PATH=$PATH:$HOME/.local/bin:$HOME/bin
PATH=$HOME/.pyenv/shims:$PATH
PATH=$HOME/.pyenv/bin:$PATH
export PATH

eval "$(pyenv init -)"

pyenv install 3.10.2
pyenv global 3.10.2

curl -sSL https://install.python-poetry.org | python3 - --uninstall
curl -sSL https://install.python-poetry.org | python3 -
PATH=$HOME/.poetry/bin:$PATH
export PATH
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

if [ ! -d $HOME/Repos ]; then
   mkdir $HOME/Repos
fi

if [ ! -d $HOME/Repos/lab-tools ]; then
   git clone https://github.com/Barmada-Lab/lab-tools $HOME/Repos/lab-tools
fi

$HOME/Repos/lab_tools/scripts/update.sh
