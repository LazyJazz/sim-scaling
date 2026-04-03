import torch
import numpy as np
import sim_scaling.policy.base_policy
import sim_scaling.manager.base_manager
import sim_scaling.task.base_env
import time

from pxr import Usd, UsdGeom
import pxr
import omni

class USDManipManager(sim_scaling.manager.base_manager.BaseManager):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.exit_flag = False
        self.run_steps = 0
        self.stage = self.env.stage
        self.current_prim = None

    def step(self, obs, action, step_result=False):
        super().step(obs, action, step_result)

        if self.run_steps > 0:
            self.run_steps -= 1
            return

        input_string = input("Enter command (or 'exit' to quit): ")
        if input_string.strip().lower() == "exit":
            self.exit_flag = True
        
        terms = input_string.split()
        cmd = terms[0] if terms else ""
        if cmd == "run":
            if len(terms) > 1:
                try:
                    self.run_steps = int(terms[1])
                except ValueError:
                    print("Invalid number of steps. Please enter an integer.")
            else:
                print("Please specify the number of steps to run.")
        if cmd == "get":
            path = terms[1] if len(terms) > 1 else ""
            prim = self.stage.GetPrimAtPath(path)
            if not prim:
                print(f"No prim found at path: {path}")
            else:
                print(f"Type of prim at {path}: {prim.GetTypeName()}, {type(prim)}")
        if cmd == "select":
            path = terms[1] if len(terms) > 1 else ""
            prim = self.stage.GetPrimAtPath(path)
            if not prim:
                print(f"No prim found at path: {path}")
            else:
                self.current_prim = prim
                print(f"Selected prim at {path}")
        if cmd == "attrs":
            if self.current_prim is None:
                print("No prim selected. Use 'select <path>' to select a prim.")
            else:
                attrs = self.current_prim.GetAttributes()
                print(f"Attributes of prim {self.current_prim.GetPath()}:")
                for attr in attrs:
                    print(f" - {attr.GetName()}: {attr.Get()}, type: {type(attr.Get())}")
        if cmd == "attr":
            if self.current_prim is None:
                print("No prim selected. Use 'select <path>' to select a prim.")
            else:
                attr_name = terms[1] if len(terms) > 1 else ""
                attr = self.current_prim.GetAttribute(attr_name)
                if not attr:
                    print(f"No attribute named '{attr_name}' found on prim {self.current_prim.GetPath()}.")
                else:
                    print(f"Value of attribute '{attr_name}': {attr.Get()}, type: {type(attr.Get())}")
        if cmd == "attr_set":
            if self.current_prim is None:
                print("No prim selected. Use 'select <path>' to select a prim.")
            else:
                attr_name = terms[1] if len(terms) > 1 else ""
                value = terms[2:] if len(terms) > 2 else ""
                value = " ".join(value)
                attr = self.current_prim.GetAttribute(attr_name)
                if not attr:
                    print(f"No attribute named '{attr_name}' found on prim {self.current_prim.GetPath()}.")
                else:
                    import ast

                    type_of_attr = type(attr.Get())
                    value = ast.literal_eval(value)
                    if type_of_attr == pxr.Gf.Quatd:
                        value = pxr.Gf.Quatd(*value)
                    elif type_of_attr == pxr.Gf.Vec3d:
                        value = pxr.Gf.Vec3d(*value)
                    elif type_of_attr == pxr.Gf.Vec3f:
                        value = pxr.Gf.Vec3f(*value)
                    elif type_of_attr == pxr.Gf.Vec2f:
                        value = pxr.Gf.Vec2f(*value)
                    elif type_of_attr == pxr.Gf.Vec2d:
                        value = pxr.Gf.Vec2d(*value)

                    attr.Set(value)
                    print(f"Set attribute '{attr_name}' to {value} on prim {self.current_prim.GetPath()}.")
        
        if cmd == "save_vis":
            imgs = obs['rgb']
            b, h, w, c = imgs.shape
            for i in range(b):
                path = f"imgs/"
                # create directory if not exists
                import os
                os.makedirs(path, exist_ok=True)
                img = imgs[i].cpu().numpy()
                from PIL import Image
                im = Image.fromarray(img)
                im.save(f"{path}/{i}.png")
            print(f"Saved {b} images to {path}")
        

    def should_terminate(self):
        return self.exit_flag
    
    def __repr__(self):
        return f"Iter.{self.iter}: success_count: {self.env.success_count}, done_count: {self.env.done_count}, success_rate: {self.env.get_success_rate():.3f}"