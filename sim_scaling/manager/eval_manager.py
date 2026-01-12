import sim_scaling.manager.base_manager
import sim_scaling.task.base_env

import os
import json


class EvalManager(sim_scaling.manager.base_manager.BaseManager):
    def __init__(self, save_path, num_eval_envs=1000, **kargs):
        super().__init__(**kargs)
        print(f"Init seed: {self.env.init_seed}")
        self.save_path = save_path
        self.num_eval_envs = num_eval_envs
        self.success_records = [None] * num_eval_envs
        self.success_count = 0
        self.record_count = 0

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        

    def step(self, obs, action):
        super().step(obs, action)
        while self.env.done_queue:
            done_event = self.env.done_queue.pop(0)
            seed = done_event["seed"]
            if seed < self.num_eval_envs + self.env.init_seed:
                self.success_records[seed - self.env.init_seed] = done_event["success"]
                if done_event["success"]:
                    self.success_count += 1
                self.record_count += 1


    def should_terminate(self):
        if self.record_count >= self.num_eval_envs:
            save_file = os.path.join(self.save_path, f"success.json")
            with open(save_file, "w") as f:
                json.dump(self.success_records, f)
            save_file = os.path.join(self.save_path, f"record.json")
            with open(save_file, "w") as f:
                json.dump(self.env.done_record, f)
            save_file = os.path.join(self.save_path, f"success_rate.json")
            with open(save_file, "w") as f:
                json.dump({
                    "success_count": self.success_count,
                    "record_count": self.record_count,
                    "success_rate": self.success_count / max(1, self.record_count)
                }, f)
            return True
        return False

    def __repr__(self):
        return f"Iter.{self.iter}: eval_envs_count: {self.record_count}/{self.num_eval_envs}, success_count: {self.success_count}, success_rate: {self.success_count / max(1, self.record_count):.3f}" 