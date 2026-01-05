import task.base_env

class PushTEnv(task.base_env.BaseEnv):
    def __init__(self):
        super().__init__()

    def reset(self, idx=None):
        return "PushTEnv reset called"
    