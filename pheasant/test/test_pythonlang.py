'''
Pythonの言語使用を学ぶためのスクリプトです。
参考:
https://docs.python.org/ja/3/tutorial/index.html
'''
from datetime import datetime

def test_gettime():
  now = datetime.now()
  print(now)
  assert now != None

def test_slice_str():
  word = 'JavaScript'
  w = word[4:]
  assert w == 'Script'

def test_range_to_list():
  lst = list(range(0, 5))
  assert lst == [0,1,2,3,4]

def test_enumerate_list():
  lst = list(range(1,11))
  n = 0
  for index, value in enumerate(lst):
    n += value
  assert n == 55

def test_range_with_sum():
  assert sum(range(1, 11)) == 55

class TestMember:
  def __init__(self, name, age):
    self.name = name
    self.age = age

def test_match_sentense(): 
  mem = TestMember('Mike', 21)
  result = None
  match mem:
    case TestMember(name=name, age=age) if age >= 18:
      result = 'Adult'
    case _:
      result = 'Child'
  assert result == 'Adult'

def test_default_function_args():
  #デフォルト引数は後続の関数と共有される。事故を避けたければデフォルト引数には不変な値を指定する。
  def addlist(v, list=[]):
    list.append(v)
    return list

  addlist('foo')
  addlist('bar')
  l = addlist('baz')

  assert ','.join(l) == 'foo,bar,baz'

def test_dict_args():
  def sample(**kv):
    res = []
    for key in kv: #key, valueでイテレートできない。
      print(key)
      res.append(key + '=' + kv[key])
    return res
  
  #値が数値だとValueErrorになる。
  result = sample(Math='90', English='78', Music='87')

  assert ','.join(result) == 'Math=90,English=78,Music=87'
