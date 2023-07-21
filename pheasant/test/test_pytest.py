'''
PyTestの動作確認を行うためのスクリプトです。
'''
from datetime import datetime

def test_gettime():
  now = datetime.now()
  print(now)
  assert now != None
