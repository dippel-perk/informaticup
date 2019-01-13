import string
import sys
import time
from colorama import init,Fore,AnsiToWin32

init(autoreset=True)

error_stream = AnsiToWin32(sys.stderr).stream

def print_countdown(wait_time: int, prefix_text: string = "Waiting for"):

    assert wait_time < 100

    countdown_message = "\r" + prefix_text + " %2d seconds"

    for remaining in range(wait_time, 0, -1):
        sys.stdout.write(countdown_message % (remaining))
        sys.stdout.flush()
        time.sleep(1)

    sys.stdout.write("\r" + " " * len(countdown_message) + "\r")

def print_error(message: string):
    print(Fore.RED + message, file=error_stream)

def print_debug(message: string):
    print(Fore.BLUE + message)

def print_success(message: string):
    print(Fore.GREEN + message)
