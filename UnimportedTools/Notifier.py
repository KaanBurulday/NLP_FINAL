from win10toast import ToastNotifier

def notify_complete():
    toaster = ToastNotifier()
    toaster.show_toast("PyCharm Notification", "Your script has completed!", duration=2)


def notify_message(title, message, duration=2):
    toaster = ToastNotifier()
    toaster.show_toast(title, message, duration)
