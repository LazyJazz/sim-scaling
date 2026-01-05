import task.pusht

if __name__ == "__main__":
    env = task.pusht.PushTEnv()
    # while env.app.is_running():
    for i in range(100):
        env.step()
    env.close()

    print("Finished running PushTEnv example.")