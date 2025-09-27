"""
PyOTPをテストします。

参考:
https://pypi.org/project/pyotp/
"""
import pyotp
import time

def test_verify_pyotp():
  """
  PyOTPで作成したワンタイムパスワードの有効期限に関する振る舞いをテストします。
  """
  secret_key = pyotp.random_base32()
  # テストのため1秒でパスワードの有効期限が切れるようにする。規定値は30秒である。
  totp = pyotp.TOTP(secret_key, interval=1) 
  password = totp.now()

  # 有効期限が切れていないことのテスト
  assert totp.verify(password)
  time.sleep(1)
  
  # 有効期限が切れていることのテスト
  assert not totp.verify(password)

