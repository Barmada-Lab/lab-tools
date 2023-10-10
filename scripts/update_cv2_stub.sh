# opencv doesn't ship with a method signature stub, and pyright can't auto-generate
# a stub since opencv isn't written in python. If pyright complains and won't give
# you type information, you probably need to download the stub to your cv2 source
# folder:

curl -sSL https://raw.githubusercontent.com/microsoft/python-type-stubs/main/cv2/__init__.pyi \
    -o $(poetry run python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')/__init__.pyi
