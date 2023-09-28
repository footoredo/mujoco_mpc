import sys

from core import Runner

if __name__ == "__main__":
    env = sys.argv[1]
    runner = Runner(env, use_viewer=True, save_video=False, save_last_img=True, init_data=None)
    runner.execute(custom=False, reset_after_done=False)
    runner.finish()
