# docker trainer image

docker trainer image contains trainer code

## build docker image
### Usage
````bash
bash ./build_image.sh [cpu|gpu] [version]
````

* `[cpu|gpu]` specific which type of tensorflow will used, default value of this option is `cpu`.
   Note that `gpu` type of docker image need `nvidia-docker` to run.
* `[version]` specific docker image label, default value is `0.0.1`

NOTE: final output docker image name is `ner_base-{arch}:{version}` which `{arch}` is the value of first option, `{version}` is the value of second option.

### example
bash script:
```bash
bash ./build_image.sh
```

You will get Docker image: `ner_trainer-cpu:0.0.1` in which `-cpu` indicate this image is CPU based.

## run docker image
### Usage
```bash
bash ./run_image.sh [data_dir] [config_file]
```

see example at `run_image.sh` for how mount model dir and specific http port


## test docker
see example at `test_docker.sh`
