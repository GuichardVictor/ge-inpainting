# PFEE - GE Healthcare - Inpainting

## Authors

* Victor Guichard
* Guillaume Valente
* Mehdi Bekhtaoui
* Alexandre Yvart

## Content

### Data

2D Generators:

```python
from inpainting.data import CustomGenerator, MaskGenerator
```

2D+t Patch Generation

```python
from inpainting.data import generate_patches
```

2D+t Mask Generation (creates a volume)

```python
from inpainting.data import Generate3DArtifacts
```

### Layers

This package contains the implementation of Partial Convolution 2D and Partial Convolution 3D

```python
from inpainting.layers import PConv2D, PConv3D
```

These layers are similar to Conv2D, Conv3D but take as input the image and the mask:

```python
PConv3D(filters, kernel_size, padding='same')([img, mask])
PConv2D(filters, kernel_size, padding='same')([img, mask])
```

### Models

A pre defined model for PConv2D and PConv3D (many2many and many2one) are implemented:

```python
from inpainting.model import PConvModel, PConvModel3D

model = PConvModel(...) # Pre compiled and ready to use 2D model
model = PConvModel3D(...) # Pre compiled and ready to use 3D many2many model
model = PConvModel3D(..., many2many=False) # Pre compiled and ready to use 3D many2one model
```

### Losses

The custom vgg loss are defined inside the models

### Data

* Custom Generator: Custom Keras DataGenerator that will output image mask and ground truth
* Mask Generator: Generator that build 2D masks

## Contact

If some parts are unclear you can contact: [Victor Guichard](mailto:guichardvictor@gmail.com?subject=[PFEE][HELP])