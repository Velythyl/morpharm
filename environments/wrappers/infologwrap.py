import gymnasium


class InfoLogWrap(gymnasium.Wrapper):
    def __init__(self, env, prefix):
        super().__init__(env)
        self.prefix = prefix

    def step(self, action):
        ret = self.env.step(action)
        new_info = {}
        for key, value in ret[-1].items():
            new_info[f"{self.prefix}#{key}"] = value
        new_info[f"{self.prefix}#rew"] = ret[1]
        return *ret[:-1], new_info
