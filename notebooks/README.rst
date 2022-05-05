=========
Notebooks
=========

Hey look it's a collection of somewhat user-friendly scripts,, that's pretty cool


If you wanna use these on greatlakes, all you need to do is the following:

1. Make sure you have pyenv, python 3.9.9, and poetry installed

2. Clone this repository:

.. code:: bash

    $ mkdir ~/Repos && cd ~/Repos
    $ git clone https://github.com/kerowak/lab_tools

3. Do this stuff

.. code:: bash

    $ cd ~/Repos/lab_tools/notebooks
    $ poetry install
    $ poetry run ./nbkgen.sh

4. Start up jupyterlab from the greatlakes dashboard, et voila!
