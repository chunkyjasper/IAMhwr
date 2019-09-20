"""
Basic publisher subscriber pattern
"""

subscribers = {}


def unsub(event, callback):
    if event is not None or event != "" \
            and event in subscribers.keys():
        subscribers[event] = list(
            filter(
                lambda x: x is not callback,
                subscribers[event]
            )
        )


def unsub_all():
    global subscribers
    subscribers = {}


def sub(event, callback):
    if not callable(callback):
        raise ValueError("callback must be callable")

    if event is None or event == "":
        raise ValueError("Event cant be empty")

    if event not in subscribers.keys():
        subscribers[event] = [callback]
    else:
        subscribers[event].append(callback)


def pub(event, args):
    if event in subscribers.keys():
        for callback in subscribers[event]:
            callback(args)
