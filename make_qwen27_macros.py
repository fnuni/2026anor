"""Generate the Qwen-27B pilot macros (qwen27_macros.tex) from data/qwen27_pilot/."""
from pilot_macros import QwenLargeMacros


def main():
    return QwenLargeMacros().write()


if __name__ == '__main__':
    main()
