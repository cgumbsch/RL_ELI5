"""Static picture of the grid world with the robot at the start.

    python -m demos.world [--layout small|large]
"""
import argparse

import gridworld as gw
import render as rd


def main():
    a = rd.add_common_args(argparse.ArgumentParser(description=__doc__), "world").parse_args()
    env = gw.GridWorld(a.layout)
    cv = rd.Canvas(env, a.width, a.height)
    rd.draw_grid(cv)
    rd.draw_robot(cv, env.start)
    print("wrote", cv.save(f"{a.out}.png"))
    cv.close()


if __name__ == "__main__":
    main()
