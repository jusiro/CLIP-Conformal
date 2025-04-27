We suggest the following dataset organization to ease management and avoid modifying the source code.
The datasets structure looks like:

```
CLIP-Conformal/
└── local_data/
    └── datasets/
        ├── caltech-101
        ├── dtd
        ├── eurosat
        ├── fgvc_aircraft
        ├── food-101
        ├── imagenet
        ├── imagenet-adversarial
        ├── imagenet-rendition
        ├── imagenet-sketch
        ├── imagenetv2
        ├── oxford_flowers
        ├── oxford_pets
        ├── stanford_cars
        ├── sun397
        └── ucf101
```

In the following, we provide specific download links and expected structure for each individual dataset. You can find
the ```classname.txt``` file for imagenet shifts at ```./imagenet/```. For other datasets, train/text splits are located at 
```./splits/```. You should paste such files into its corresponding dataset's folders. These split files are originally from 
[CoOp](https://github.com/KaiyangZhou/CoOp/)'s work (Thanks!). You may want to take a look to its repository ([LINK](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)) 
for getting detailed dataset links.

### caltech-101

```
.
└── caltech-101/
    ├── 101_ObjectCategories/
    │   ├── accordion/
    │   │   ├── image_0001.jpg
    │   │   ├── image_0002.jpg
    │   │   ├── image_0003.jpg
    │   │   ├── image_0004.jpg
    │   │   └── ...
    │   ├── airplanes/
    │   │   └── ...
    │   ├── anchor/
    │   │   └── ...
    │   └── ...
    └── split_zhou_Caltech101.json
```

### dtd

```
.
└── dtd-101/
    ├── images/
    │   ├── banded/
    │   │   ├── banded_0002.jpg
    │   │   ├── banded_0004.jpg
    │   │   ├── banded_0005.jpg
    │   │   └── ...
    │   ├── blotchy/
    │   │   └── ...
    │   ├── braided/
    │   │   └── ...
    │   ├── bubbly/
    │   │   └── ...
    │   └── ...
    ├── imdb/
    │   └── ...
    ├── labels/
    │   └── ...
    └── split_zhou_DescribableTextures.json
```

### eurosat

```
.
└── eurosat/
    ├── 2750/
    │   ├── AnnualCrop/
    │   │   ├── AnnualCrop_1.jpg
    │   │   ├── AnnualCrop_2.jpg
    │   │   ├── AnnualCrop_3.jpg
    │   │   ├── AnnualCrop_4.jpg
    │   │   └── ...
    │   ├── Forest/
    │   │   └── ...
    │   ├── HerbaceousVegetation/
    │   │   └── ...
    │   ├── Highway/
    │   │   └── ...
    │   └── ...
    └── split_zhou_EuroSAT.json
```

### fgvc-aircraft

```
.
└── fgvc-aircraft/
    ├── images/
    │   ├── 0034309.jpg
    │   ├── 0034958.jpg
    │   ├── 0037511.jpg
    │   ├── 0037512.jpg
    │   └── ...
    ├── variants.txt
    ├── images_variant_test.txt
    ├── images_variant_train.txt
    ├── images_variant_val.txt
    └── ...
```

### food-101

```
.
└── food-101/
    ├── images/
    │   ├── apple_pie/
    │   │   ├── 134.jpg
    │   │   ├── 21063.jpg
    │   │   ├── 23893.jpg
    │   │   ├── 38795.jpg
    │   │   └── ...
    │   ├── baby_back_ribs/
    │   │   └── ...
    │   ├── baklava/
    │   │   └── ...
    │   ├── beef_carpaccio/
    │   │   └── ...
    │   └── ...
    ├── meta/
    │   └── ...
    ├── license_agreement.txt
    ├── README.txt
    └── split_zhou_Food101.json
```

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

### oxford_flowers

```
.
└── oxford_flowers/
    ├── jpg/
    │   ├── image_00001.jpg
    │   ├── image_00002.jpg
    │   ├── image_00003.jpg
    │   ├── image_00004.jpg
    │   ├── image_00005.jpg
    │   └── ...
    ├── cat_to_name.json
    ├── imagelabels.mat
    └── split_zhou_OxfordFlowers.json
```

### oxford_pets

```
.
└── oxford_pets/
    ├── annotations/
    │   └── ...
    ├── images/
    │   ├── Abyssinian_1.jpg
    │   ├── Abyssinian_2.jpg
    │   ├── Abyssinian_3.jpg
    │   ├── Abyssinian_4.jpg
    │   ├── Abyssinian_5.jpg
    │   └── ...
    └── split_zhou_OxfordPets.json
```

### stanford_cars

```
.
└── stanford_cars/
    ├── cars_train/
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   ├── 00003.jpg
    │   ├── 00004.jpg
    │   ├── 00005.jpg
    │   └── ...
    ├── cars_test/
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   ├── 00003.jpg
    │   ├── 00004.jpg
    │   ├── 00005.jpg
    │   └── ...
    ├── car_devkit/
    │   └── ...
    ├── split_zhou_StanfordCars.json
    └── cars_annos.mat
```

### sun397

```
.
└── sun397/
    ├── SUN397/
    │   ├── a/
    │   │   ├── abbey/
    │   │   │   ├── sun_aaalbzqrimafwbiv.jpg
    │   │   │   ├── sun_aaaulhwrhqgejnyt.jpg
    │   │   │   ├── sun_aacphuqehdodwawg.jpg
    │   │   │   └── ...
    │   │   ├── airplane_cabin/
    │   │   │   └── ...
    │   │   ├── airport_terminal/
    │   │   │   └── ...
    │   │   └── ...
    │   ├── b/
    │   │   └── ...
    │   ├── c/
    │   │   └── ...
    │   └── ...
    ├── ClassName.txt
    ├── split_zhou_SUN397.json
    └── ...
```

### ucf101

```
.
└── ucf101/
    ├── UCF-101-midframes/
    │   ├── Apply_Eye_Makeup/
    │   │   ├── v_ApplyEyeMakeup_g01_c01.jpg
    │   │   ├── v_ApplyEyeMakeup_g01_c02.jpg
    │   │   ├── v_ApplyEyeMakeup_g01_c03.jpg
    │   │   └── ...
    │   ├── Apply_Lipstick/
    │   │   └── ...
    │   ├── Archery
    │   └── ...
    └── split_zhou_UCF101.json
```