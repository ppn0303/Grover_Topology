# this is based on GROVER (https://github.com/tencent-ailab/grover)

# changed info
1. multi-gpu changed from horovod to torch ddp
2. add multi-class classification
3. add random search & grid search in code

# docker
- I provide docker image
- docker pull ppn0303/ai_tox:fourth

# notice
- I combine MGSSL(https://github.com/zaixizhang/MGSSL)'s pretrain with GROVER's pretrain.
- but performance changed little.
