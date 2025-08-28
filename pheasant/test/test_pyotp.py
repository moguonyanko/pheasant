"""
PyOTPをテストします。

参考:
https://pypi.org/project/pyotp/
"""
import pyotp
import time

totp = pyotp.TOTP('base32secret3232')
totp.now() # => '492039'

# OTP verified for current time
totp.verify('492039') # => True
time.sleep(30)
totp.verify('492039') # => False
