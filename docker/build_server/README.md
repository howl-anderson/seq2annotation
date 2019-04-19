# docker trainer image

docker trainer image contains server code

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

You will get Docker image: `ner-cpu:0.0.1` in which `-cpu` indicate this image is CPU based.


## run docker image
see example at `run_image.sh` for how mount model dir and specific http port


## test docker
see example at `test_docker.sh`
