import termcolor

# Cyan = Debug messages
# Yellow = Warnings
# Red = Critical errors

# --------------------------------------------------------------------------
# ENABLING COLOURS ON WINDOWS 10 (from: https://bugs.python.org/issue29059 )
# --------------------------------------------------------------------------
import platform
if platform.system().lower() == 'windows':
    from ctypes import windll, c_int, byref
    stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
    mode = c_int(0)
    windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
    mode = c_int(mode.value | 4)
    windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)
# --------------------------------------------------------------------------

def warning(msg, err=None):
    if err:
        m = "Warning: "+str(msg)+"; "+type(err).__name__+": "+str(err)
    else:
        m = "Warning: "+str(msg)+"."
    termcolor.cprint(m, "yellow")

def error(msg, err=None, exit_code=1):
    if err:
        m = "Error: "+str(msg)+"; "+type(err).__name__+": "+str(err)
    else:
        m = "Error: "+str(msg)+"."
    termcolor.cprint(m, "red")
    exit(exit_code)

def timer(f, *args, **kwargs):
    import timeit
    start = timeit.default_timer()
    freturn = f(*args, **kwargs)
    end = timeit.default_timer()
    termcolor.cprint("Time taken to run "+str(f.__name__)+": "+str(end - start)+" seconds.","cyan")
    return freturn