import threading

import torch


def test_forward_mode_jvp_calls_are_serialized_across_threads():
    from nlp_solver.utils import locked_jvp
    errors = []

    def work():
        try:
            for _ in range(50):
                x = torch.ones(3, dtype=torch.float64)
                assert torch.allclose(locked_jvp(lambda z: z.sin(), (x,), (x,))[1], x.cos())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
