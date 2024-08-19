## INSTALL

1. Install `requirements.txt`
2. `cd environments/customenv/mujococustom && git clone https://github.com/google-deepmind/mujoco_menagerie.git assets`

## RENDERING

You might need to use an `LD_PRELOAD` trick if you're using conda. Here: [https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35](https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35)

Also note that we're not doing anything to the underlying simulators, so videos might look drastically different for brax vs mujoco vs PyBullet, etc.