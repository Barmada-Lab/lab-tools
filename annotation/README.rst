==========
Annotation
==========

This package provides utilities for annotating images and a cli tool for utilizing those utilities.

Installation
------------

To install this tool:

.. code:: bash

    $ cd /path/to/lab_tools
    $ pipx install annotation/

To verify its installation:

.. code:: bash

    $ anno -h
    usage: anno [-h] {convert} ...

    annotation utility

    positional arguments:
    {convert}
        convert   convert annotations from one format to another

    optional arguments:
    -h, --help  show this help message and exit

=====
Usage
=====

convert
-------

The 'convert' subcommand will convert between roi formats. Supported formats include:

1. JSON (masa export format)
2. Pickle (standard anno format)
3. IJ (NIH ImageJ roi format)

To convert an individual json roi to ij format, the syntax would be:

.. code:: bash

   $ anno convert input.json . -e ij

'convert' operates on a list of one or more inputs and drops its outputs into the destination directory. You can use glob expansions to select lists of files, if your shell supports it. For example,

.. code:: bash

   $ anno convert input_dir/*.json . -e ij

will select all files matching *.json from input_dir/, and export them to corresponding IJ rois.
