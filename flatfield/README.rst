=========
Flatfield
=========

This tool will:
1) Generate a median image from a set of inputs
2) Subtract a given image from a set of inputs

Combined together, these functions can be used to perform flatfield correction on a set of images.

------------
Installation
------------

You can install this tool using pipx. The cli tool 'flatfield' will then be made available for use.

-----
Usage
-----

To generate background images:

.. code-block::

    flatfield generate -r /PATH/TO/IMAGES /PATH/TO/OUTPUT.tif

To subtract a background image from a set of images:

.. code-block::

    flatfield subtract /PATH/TO/BACKGROUND /PATH/TO/IMAGES /OUTPUT/PATH

-----
Guide
-----

To perform flatfield correction, first select a set of images that all share common optical properties. Mainly, there should be a uniform background profile or artifact present in every image. Variations in the intensity of the background, such as those that might appear when using variable exposure lengths, will generate sub-optimal background images.

Now gather your images into a directory, and decide where you would like the generated background image to be placed. Run `flatfield generate`, specifying the -r (recursive) flag.

Once you've generated an image, you can run `flatfield subtract` against the same set of images to get a flatfielded dataset.

More images == more better, but how many you actually need will depend on how sparse your images are (how much of each individual image is "information," as opposed to background.) I've found that 128 images is pretty good for sparse flouresence microscopy data, when most of the information in an image is localized to a few neurons with varying positions from image to image. Dense images, such as when a substrate is stained with flourescent markers and appears dimly in the background of every image, require more images to construct a smooth background.
