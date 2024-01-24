import sys

from core import Runner, set_runner, set_env

if __name__ == "__main__":
    env = sys.argv[1]
    runner = Runner(env, use_viewer=True, save_video=False, save_last_img=True, init_data=None)
    set_runner(runner)
    set_env(env)
    runner.execute(custom=True, reset_after_done=False)
    runner.finish()
