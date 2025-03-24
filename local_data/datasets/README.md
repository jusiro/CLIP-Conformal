We suggest the following dataset organization to ease management and avoid modifying the source code.
The datasets structure looks like:

```
Conformal-VLMs/
└── local_data/
    └── datasets/
        ├── imagenet
        ├── imagenetv2
        ├── imagenet-sketch
        ├── imagenet-adversarial
        ├── imagenet-rendition
        ├── sun397
        ├── fgvc_aircraft
        ├── eurosat
        ├── stanford_cars
        ├── food-101
        ├── oxford_pets
        ├── oxford_flowers
        ├── caltech-101
        ├── dtd
        └── ucf101
```

In the following, we provide specific download links and expected structure for each individual dataset.

### imagenet

```
.
└── imagenet/
    ├── images/
    │   └── val/
    │       ├── n01440764/
    │       │   ├── ILSVRC2012_val_00000293.JPEG
    │       │   ├── ILSVRC2012_val_00002138.JPEG
    │       │   └── ...
    │       ├── n01440764
    │       └── ...
    └── classnames.txt
```

### imagenetv2

```
.
└── imagenetv2/
    ├── imagenetv2-matched-frequency-format-val/
    │   ├── 0/
    │   │   ├── 7e4a8987a9a330189cc38c4098b1c57ac301713f.jpeg
    │   │   ├── 20d7af22665b7749158b7eb9fa3826e.jpeg
    │   │   └── ...
    │   ├── 1
    │   ├── 2
    │   ├── 3
    │   └── ...
    └── classnames.txt
```

### imagenet-sketch

```
.
└── imagenet-sketch/
    ├── images/
    │   ├── n01440764/
    │   │   ├── sketch_1.JPEG
    │   │   ├── sketch_3.JPEG
    │   │   ├── sketch_4.JPEG
    │   │   └── ...
    │   ├── n01443537
    │   └── ...
    └── classnames.txt
```

### imagenet-adversarial

```
.
└── imagenet-adversarial/
    ├── imagenet-a/
    │   ├── n01498041/
    │   │   ├── 0.000116_digital clock _ digital clock_0.865662.jpg
    │   │   ├── 0.000348_chameleon _ box turtle_0.55540705.jpg
    │   │   └── ...
    │   ├── n01531178
    │   └── ...
    └── classnames.txt
```

### imagenet-rendition

```
.
└── imagenet-rendition/
    ├── imagenet-r/
    │   ├── n01443537/
    │   │   ├── art_0.jpg
    │   │   ├── ...
    │   │   ├── cartoon_0.jpg
    │   │   └── ...
    │   ├── n01484850
    │   └── ...
    └── classnames.txt
```