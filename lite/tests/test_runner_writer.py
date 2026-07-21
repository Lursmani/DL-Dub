import queue

from gui.runner import _QueueWriter


def drain(q):
    ops = []
    while not q.empty():
        ops.append(q.get_nowait())
    return ops


def test_plain_lines_append():
    q = queue.Queue()
    w = _QueueWriter(q)
    w.write("hello\nworld\n")
    assert drain(q) == [("append", "hello"), ("append", "world")]


def test_carriage_return_becomes_replace():
    # the autodub poll status redraws itself with \r — it must be ONE
    # self-updating log line
    q = queue.Queue()
    w = _QueueWriter(q)
    w.write("status: dubbing… (5s)\rstatus: dubbing… (10s)\r")
    ops = drain(q)
    assert ops == [("append", "status: dubbing… (5s)"),
                   ("replace", "status: dubbing… (10s)")]
