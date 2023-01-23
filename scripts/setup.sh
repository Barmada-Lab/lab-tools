#!/bin/bash
#
set -e

TURBO=/nfs/turbo/umms-sbarmada
SCRATCH=/scratch/sbarmada_root/sbarmada0/$USER

if [ ! -e $HOME/turbo ]; then
    ln -s $TURBO $HOME/turbo
fi    

if [ ! -e $HOME/scratch ]; then
    ln -s $SCRATCH $HOME/scratch
fi    

if [ ! -e $HOME/Desktop/turbo ]; then
    ln -s $TURBO $HOME/Desktop/turbo
fi    

if [ ! -e $HOME/Desktop/scratch ]; then
    ln -s $SCRATCH $HOME/Desktop/scratch
fi    

if [ ! -e $HOME/Desktop/ImageJ2.desktop ]; then
    cp $TURBO/shared/Fiji.app/ImageJ2.desktop $HOME/Desktop
fi


if [ -d $HOME/.pyenv ]; then
    rm $HOME/.pyenv -rf
fi

git clone https://github.com/pyenv/pyenv $HOME/.pyenv

idem_patch_bashprofile() {
    profile=$HOME/.bash_profile
    if ! grep -Fxq "$1" $profile; then
        echo "$1" >> $profile
    fi
    eval $1
}

idem_patch_bashprofile 'export PATH=$PATH:$HOME/.local/bin:$HOME/bin'
idem_patch_bashprofile 'export PATH=$HOME/.pyenv/shims:$PATH'
idem_patch_bashprofile 'export PATH=$HOME/.pyenv/bin:$PATH'
idem_patch_bashprofile 'eval "$(pyenv init -)"'

pyenv install 3.10.6
pyenv global 3.10.6

if [ -d $HOME/.poetry ]; then
    curl -sSL https://install.python-poetry.org | python3 - --uninstall
fi

curl -sSL https://install.python-poetry.org | python3 -
idem_patch_bashprofile 'export PATH=$HOME/.poetry/bin:$PATH'
idem_patch_bashprofile 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring'

#if [ ! -d $HOME/Repos ]; then
#   mkdir $HOME/Repos
#fi
#
#LABTOOLS=$HOME/Repos/lab-tools
#cp $TURBO/shared/.bash_profile $HOME
#
#if [ ! -d $LABTOOLS ]; then
#   git clone https://github.com/Barmada-Lab/lab-tools $LABTOOLS
#fi
#
#$LABTOOLS/scripts/update.sh
