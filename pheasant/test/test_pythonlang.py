'''
Pythonの言語使用を学ぶためのスクリプトです。
参考:
https://docs.python.org/ja/3/tutorial/index.html
'''
from datetime import datetime as dt
import os
from dataclasses import dataclass

def test_gettime():
  now = dt.now()
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

def test_args_keywords():
  #/の前にあるnameは位置専用引数、*の後ろにあるmessageはキーワード専用引数
  def sample(name, /, id, *, message):
    return '_'.join([name, id, message])

  #sample(name='Mike', 'A01', message='Hello') #シンタックスエラー
  #sample('Mike', id='A01', 'Hello') #シンタックスエラー
  #sample('Mike', 'A01', 'Hello') #実行時エラー（TypeError）
  result = sample('Mike', 'A01', message='Hello')
  assert result == 'Mike_A01_Hello'

def test_lambda():
  def create_hello():
    return lambda name: 'Hello,' + name
  f = create_hello()
  assert f('Taro') == 'Hello,Taro'

def test_lambda_with_sort():
  members = [('Taro', 21), ('Masao', 18), ('Mike', 43), ('Joe', 28), ('Usao', 30)]
  #タプルの第2要素（年齢）でソート
  members.sort(key=lambda member: member[1]) 
  print(members)
  assert members[0] == ('Masao', 18)

def test_func_with_types():
  def hello(name: str) -> str:
    print(hello.__annotations__)
    return 'Hello,' + name + '!'
  assert hello('Masao') == 'Hello,Masao!'

def test_list_naihou_hyouki():
  result = [(x, x**2) for x in range(10) if x % 2 == 0 and x > 0]
  assert result == [(2, 2**2), (4, 4**2), (6, 6**2), (8, 8**2)]

def test_pack_to_tuple():
  t = 1, 10, 100
  assert t == (1, 10, 100)

def test_unpack_from_tuple():
  t = (100, 200, 300)
  a, b, c = t
  assert a + b + c == 600

def test_jisho_naihou_hyouki():
  #inの後ろのタプルはリストでもよい。
  d = {name: len(name) for name in ('Masao', 'Jiro', 'Daigorou')}
  assert d == {'Masao': 5, 'Jiro': 4, 'Daigorou': 8}

def test_zip_iterate():
  a = [1, 2, 3]
  b = ['Mike', 'Taro', 'Jiro']
  result = []
  for id, name in zip(a, b):
    result.append(str(id) + ':' + name)
  
  assert '_'.join(result) == '1:Mike_2:Taro_3:Jiro'

def test_formatted_string():
  name = 'Mike'
  age = 34
  result = f'name={name},age={age}'
  #以下だとname='Mike'になってしまう。ninattesimau.
  #result = f'{name=},{age=}'

  assert result == 'name=Mike,age=34'

def test_formatted_table():
  table = {
    'Mike': 34,
    'Taro': 21,
    'Joe': 45
  }
  res = 'Mike: {Mike:d}, Taro: {Taro:d}, Joe: {Joe:d}'.format(**table)

  assert res == 'Mike: 34, Taro: 21, Joe: 45'

def test_read_file():
  print(os.getcwd())
  #~でユーザーディレクトリを参照することはできない。
  with open('/usr/local/var/www/index.php', encoding='utf-8') as f:
    for line in f:
      #endに改行を指定しなくても改行はそのまま読み込まれる。
      print(line, end='')
  #withでファイルはクローズされている。
  assert f.closed == True

def test_raise_error_chaining():
  def throw_error():
    try :
      open('this_is_nothing.txt')
    except OSError as err:
      #fromを介して翻訳元の例外を指定できる。翻訳元例外は隠蔽したい時もあるが調査には便利かもしれない。
      raise RuntimeError('Cannot open file') from err

  try:  
    throw_error()
  except FileNotFoundError as fe:
    assert False
  except RuntimeError as re:
    #あくまでも送出されてくるのは翻訳後の例外である。fromを使ったからといって翻訳元の例外が来るわけではない。
    assert True

def test_throw_exceptiongroup():
  def throw_error():
    errs = [
      #両者に共通する層があるならExceptionGroupを使うのではなく
      #抽象的なクラスを定義してそれぞれに継承させた方がいい。
      OSError('e1'), ValueError('e2')
    ]  
    raise ExceptionGroup('Multi error', errs)

  try:
    throw_error()
  except Exception as e:
    assert True

  err_cnt = 0
  try:
    throw_error()
  #例外の型はexcept*でもExceptionGroupのままだがExceptionGroupに含まれる全ての例外が捕捉される。
  except* OSError as oe:
    print(f'Catched {type(oe)}')
    err_cnt += 1
  except* ValueError as ve:
    print(f'Catched {type(ve)}')
    err_cnt += 1

  assert err_cnt == 2

def test_multi_extends():
  class Base1():
    id = 1
  class Base2():
    id = 2
  class Base3():
    id = 3

  class Child(Base1, Base2, Base3): 
    def __init__(self):
      self.id = Base1.id + Base2.id + Base3.id

  assert Child().id == 6

def test_data_class():
  @dataclass
  class Member():
    name: str
    age: int

  m = Member('Mike', 45)
  assert m.name == 'Mike' and m.age == 45

def test_custom_iterater():
  class Upper:
    def __init__(self, word: str):
      self.word = word
      self.index = 0

    def __iter__(self):
      return self
    
    def __next__(self):
      if self.index == len(self.word):
        raise StopIteration
      s = self.word[self.index].upper()
      self.index += 1
      return s

  u = Upper('hello') 
  res = []
  for w in u:
    res.append(w)

  assert ''.join(res) == 'HELLO'
