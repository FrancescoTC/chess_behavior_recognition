from games_manipulation import evaluation_difference


def reward_fun_stockfish(completions, **kwargs):
    return [evaluation_difference(completion) for completion in completions]