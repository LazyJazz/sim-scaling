import task.pusht

if __name__ == "__main__":
    env = task.pusht.PushTEnv()
    env.reset()
    for i in range(100):
        obs = env.get_observations()
        print(f"Step {i}, T shape pose: {obs['t_pose']}, Target pose: {obs['targ_pose']}")
        env.step()
    env.close()

    print("Finished running PushTEnv example.")