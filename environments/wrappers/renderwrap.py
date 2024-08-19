import gymnasium


class RenderWrap(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def render(self, *args, **kwargs):
        return super(RenderWrap, self).render()
