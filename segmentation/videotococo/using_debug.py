## This file is used to debug RuntimeError
## Import this file in the testing file if there exists some unknown problem.

import sys
def info(type, value, tb):
    # 异常类型
    # 异常值
    # 调用栈
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)
sys.excepthook = info
