import ctypes
from ctypes import wintypes
    
_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

def _check_bool(result, func, args):
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    # else build final result from result, args, outmask, and 
    # inoutmask. Typically it's just result, unless you specify 
    # out/inout parameters in the prototype.
    return args

_HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

_kernel32.SetConsoleCtrlHandler.errcheck = _check_bool
_kernel32.SetConsoleCtrlHandler.argtypes = (_HandlerRoutine, 
                                            wintypes.BOOL)

_console_ctrl_handlers = {}

def set_console_ctrl_handler(handler):
    if handler not in _console_ctrl_handlers:
        h = _HandlerRoutine(handler)
        _kernel32.SetConsoleCtrlHandler(h, True)
        _console_ctrl_handlers[handler] = h
