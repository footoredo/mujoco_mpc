import sys

from core import Runner, set_runner, set_env, set_camid

if __name__ == "__main__":
    env = sys.argv[1]
    runner = Runner(env, use_viewer=False, save_video=False, save_last_img=True, init_data=None)
    set_camid(1)
    set_runner(runner)
    set_env(env)
    runner.execute(custom=False, reset_after_done=False)
    runner.finish()
