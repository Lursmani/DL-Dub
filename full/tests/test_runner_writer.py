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
    # tqdm redraws its bar with \r — the bar must be ONE self-updating line
    q = queue.Queue()
    w = _QueueWriter(q)
    w.write("10%\r20%\r30%\r")
    ops = drain(q)
    assert ops[0] == ("append", "10%")
    assert ops[1:] == [("replace", "20%"), ("replace", "30%")]


def test_partial_line_flushes_then_replaces_on_completion():
    q = queue.Queue()
    w = _QueueWriter(q)
    w.write("downloading")          # partial, no terminator yet
    w.write(" done\n")              # completed on a later write
    ops = drain(q)
    assert ops == [("append", "downloading"), ("replace", "downloading done")]


def test_blank_lines_are_kept():
    q = queue.Queue()
    w = _QueueWriter(q)
    w.write("a\n\nb\n")
    assert drain(q) == [("append", "a"), ("append", ""), ("append", "b")]
