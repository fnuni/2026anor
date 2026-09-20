"""Generate the Qwen-7B pilot macros (ollama_macros.tex) from data/ollama_pilot/."""
from pilot_macros import QwenSmallMacros


def main():
    return QwenSmallMacros().write()


if __name__ == '__main__':
    main()
