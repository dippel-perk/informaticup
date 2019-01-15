import string
import sys
import time

from colorama import init, Fore, AnsiToWin32, Style

"""
This file contains all output utilities which are needed throughout the project.
We decided to not wrap the functions into a class for readability of the resulting code.
"""

init(autoreset=True)
error_stream = AnsiToWin32(sys.stderr).stream


def print_countdown(wait_time: int, prefix_text: string = "Waiting for") -> None:
    """
    Prints a countdown, which updates itself every second. After the countdown has finished,
    the complete text will be removed.
    :param wait_time: The waiting time in seconds.
    :param prefix_text: The text which should be displayed before the current seconds.
    :return: None
    """
    assert wait_time < 100

    countdown_message = "\r" + prefix_text + " %2d seconds"

    for remaining in range(wait_time, 0, -1):
        sys.stdout.write(countdown_message % (remaining))
        sys.stdout.flush()
        time.sleep(1)

    sys.stdout.write("\r" + " " * len(countdown_message) + "\r")


def print_error(message: string) -> None:
    """
    Prints and error message.
    :param message: The error message
    :return: None
    """
    print(Fore.RED + message, file=error_stream)


def print_debug(message: string) -> None:
    """
    Prints a debug message. This function can be removed if we decide to rollout the application.
    :param message: The message.
    :return: None
    """
    print(Fore.BLUE + message)


def print_success(message: string) -> None:
    """
    Prints a success message.
    :param message: The message.
    :return: None
    """
    print(Fore.GREEN + message)


def print_info(message: string) -> None:
    """
    Prints an info message.
    :param message: The message
    :return: None
    """
    print(Fore.CYAN + "[Info]" + Fore.RESET + " " + message)


def print_variable(variable: string, value: string) -> None:
    """
    Prints a variable with its value.
    :param variable: The variable name.
    :param value: The value.
    :return: None.
    """
    print("%s %s" % (variable.ljust(25), value))


def print_space() -> None:
    """
    Prints some space so that new content can be printed out afterwards.
    :return: None
    """
    print("")


def make_bold(message: string) -> str:
    """
    Given a string, returns the string with ANSI characters so that the string gets displayed bold in the terminal.
    :param message: The message.
    :return: The bold message.
    """
    return Style.BRIGHT + message + Style.RESET_ALL
