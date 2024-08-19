## INSTALL

1. Install `requirements.txt`
2. `cd environments/customenv/mujococustom && git clone https://github.com/google-deepmind/mujoco_menagerie.git assets`

## RENDERING

You might need to use an `LD_PRELOAD` trick if you're using conda. Here: [https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35](https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35)

## THE COMPETITION

We provide a training environment that loads ten arms at once. You can use the files in `config/multienv` and `config/env` 
to control which arms are loaded, what amount of domain randomization is applied, as well as how many parallel simulators per arm are loaded.

### The Task

Code a policy along with associated training and deployment wrappers. 
You can use any number of the arms mentioned in `config/env/all-with-dr.yaml`.
You may not use the `wx250s` as part of training.
Your goal is to obtain as high performance as possible on the `wx250s`.
The task your policy must solve is to reach a target end effector position (x=0.5, y=0, z=0.3).

### Evaluation 

You are allowed to use the `wx250s` for evaluation purposes, as defined in `config/env/allandwidow-with-dr.yaml`.
Indeed, you will notice that in `config/multienv/multienv.yaml`, all arms (except `wx250s`) are defined as training environments
and all arms (including `wx250s`) are defined as evaluation environments.


### At CoRL

We will host a friendly competition. 

**If our resources permit**, a physical `wx250s` will be on premises, 
and we will deploy your policy on it.
During this deployment, the API from your policy's POV will be the same as during training: you will receive a `wx250s` 
observation with the same shape as is used in this simulator.

