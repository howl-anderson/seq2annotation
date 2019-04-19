# Basic docker image

Basic docker image contains seq2annotation package without any application code

## build docker image
### Usage
````bash
bash ./build_image.sh [cpu|gpu] [version]
````

you can specific `cpu` or `gpu` based tensorflow and specific docker image label.

NOTE: `cpu` is the default tensorflow type, and default version is `0.0.1`

### example
bash script:
```bash
bash ./build_image.sh
```

You will get Docker image: `ner_base-cpu:0.0.1` in which `-cpu` indicate this image is CPU based.

## run docker image
Add some default information print function for seq2annotation


## test docker
using HTTP client to test image if it works correctly
see example at `test_docker.sh`
