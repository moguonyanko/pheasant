'''
Pythonの言語使用を学ぶためのスクリプトです。
参考:
https://docs.python.org/ja/3/tutorial/index.html
'''
from datetime import datetime as dt
import os
from dataclasses import dataclass
import asyncio
from collections import *
import heapq
import re
from enum import Enum, IntEnum, IntFlag, Flag, auto
import json
from operator import itemgetter, attrgetter
import unicodedata
from typing import TypeAlias

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

def test_generator_formula():
  assert 55 == sum(x for x in range(1, 11))

def test_coroutine():
  async def sample():
    print('HELLO')
    #asyncio.sleepを使っているとブレークポイントで停止できない？
    await asyncio.sleep(1)
    print('WORLD')

  sample()

def test_coroutine_awaitable():
  async def plus():
    return 1

  async def inc(n: int):
    return n + await plus()

  assert asyncio.run(inc(10)) == 11

def test_coroutine_awaitable_with_task():
  # awaitの存在しない関数をasyncにすること自体は問題ない。
  # プリミティブな数値などawaitを指定できないオブジェクトにawaitを指定するとエラーになる。
  async def plus():
    return 1

  async def inc(n: int):
    task = asyncio.create_task(plus())
    return n + await task

  assert asyncio.run(inc(10)) == 11

def test_asyncio_gather():
  async def pow(n):
    return n**2
  
  async def get_gather():
    L = await asyncio.gather(
      pow(2), pow(3), pow(4)
    )
    return L

  result = asyncio.run(get_gather())
  #awaitを指定してgatherしているので結果のリストが返ってくる。結果を取得する処理は不要である。
  #assert sum(r.result() for r in result) == 29
  assert sum(result) == 2**2 + 3**2 + 4**2

def test_match_any_object():
  class Car():
    def __init__(self, name: str) -> None:
      self.name = name

  car = Car('MyCar')
  result = False
  match car.name:
    case 'MyCar':
      # オブジェクトを指定すると実行時エラーになってしまう。
    # case Car('MyCar'):
      result = True
    case _:
      result = False

  assert result == True

def test_multi_with():
  text = []

  class A():
    def __enter__(x):
      text.append('A')
    def __exit__(*args):
      text.append('a')

  class B():
    def __enter__(x):
      text.append('B')
    def __exit__(*args):
      text.append('b')

  with (A() as a, B() as b):
    pass

  assert ''.join(text) == 'ABba'

def test_create_ChainMap():
  a = {'mike': 90, 'taro': 80}
  b = {'joe': 95, 'jiro': 65}
  result = ChainMap(a, b)
  expected = {'mike': 90, 'taro': 80, 'joe': 95, 'jiro': 65}
  #ChainMapとDictで型が違うが等しいと判定される。
  assert result == expected

def test_create_Counter():
  sample = '''
    Python is a popular programming language because it is simple, readable, versatile, and supported by a large community.

    I removed some of the details in order to keep the summary concise. I also changed the order of some of the points to improve the flow.

    Here is another option:

    Python is a powerful and versatile language that is easy to learn and use.

    This is a more general summary that focuses on the key advantages of Python.
'''
  words = re.findall(r'\w+', sample)
  cnt = Counter(words)
  result = cnt.most_common(10)
  least_element = result[:-len(result)-1:-1][0]
  assert ('I', 2) == least_element

def test_create_deque():
  d = deque(range(5))
  result = []
  for ele in d:
    result.append(str(ele))
  result = ''.join(result)
  assert result == '01234'

def test_create_defaultdict():
  def default_func(v):
    return lambda: v
  d = defaultdict(default_func('Undefined'))
  d.update(name='Mike', age=34)
  result = '%(name)s is %(age)d years old, He like %(lang)s.' % d
  assert result == 'Mike is 34 years old, He like Undefined.'

def test_create_namedtaple():
  Member = namedtuple('Member', ['name', 'age'])
  m = Member('Mike', age=45)
  assert f'{m.name} is {m.age} years old' == 'Mike is 45 years old'
  _, age = m
  assert age >= 20
  assert getattr(m, 'name') == 'Mike'

def test_sequencial_dict():
  '''
  Python3.7以降辞書のキーは要素を挿入した順序で返されることが保証されるようになった。
  '''
  d = {'b': 1, 'a': 2, 's': 3, 'e': 4}
  s = ''.join(d.keys())
  assert 'base' == s

def test_format_with_list():
  names = ['Python', 'Java', 'JavaScript']
  #{}に変数が埋め込まれる。
  result = ['{} Lang'.format(name) for name in names]
  result = ','.join(result)
  assert result == 'Python Lang,Java Lang,JavaScript Lang'

def test_sort_by_heapq():
  l = [7,6,2,4,1,8,3,5]
  h = []
  for v in l:
    heapq.heappush(h, v)
  #heapqを小さい方からpopするだけで昇順にソートされた結果が得られる。（ヒープソート）
  #ただしsorted()と異なりステーブルソートではない。
  result = [heapq.heappop(h) for _ in range(len(h))]
  assert result == [1,2,3,4,5,6,7,8]

#参考
#https://docs.python.org/ja/3/howto/enum.html#using-automatic-values
class AutoWeekdayName(Enum):
  def _generate_next_value_(name, start, count, last_values):
    return name[0:3]
  
class WeekDay(AutoWeekdayName):
  SUNDAY = auto()
  MONDAY = auto()
  TUESDAY = auto()
  WEDNESDAY = auto()
  THURSDAY = auto()
  FRIDAY = auto()
  SATURDAY = auto()

def test_assign_enum_other_auto_names():
  assert WeekDay.SUNDAY.value == 'SUN'

#IntEnumやIntFlagは非推奨

def test_create_intenum():
  class Status(IntEnum):
    OK = 200
    NOT_FOUND = 404
  
  def is_ok(status: Status):
    return status <= 200

  assert is_ok(Status.OK)

def test_create_intflag():
  class Climbing(IntFlag):
    CLEAR = 10
    ZONE = 5
    FAIL = 0
    CLEAR_AND_ZONE = CLEAR + ZONE

  assert Climbing.FAIL < Climbing.CLEAR_AND_ZONE

def test_create_flag():
  class HSV(Flag):
    H = auto()
    S = auto()
    V = auto()
    BLACK = V == 0 #boolを混ぜてもエラーにならない
  
  red = HSV.H | HSV.S | HSV.V
  assert bool(red)

def test_create_custom_new_enum():
  class Mark(Enum):
    HAERT = auto()
    CLUB = auto()
    SPADE = auto()
    DIAMOND = auto()

  class Card(Enum):
    JACK = (Mark.HAERT, 11)  
    QUEEN = (Mark.DIAMOND, 12)
    KING = (Mark.SPADE, 13)

    def __init__(self, mark, number):
      self.mark = mark
      self.number = number
    @property
    def description(self):
      return f'{self.mark}:{self.number}'

  assert Card.KING.description == 'Mark.SPADE:13'

def test_dump_json():
  src = {"name": "Mike", "age": 25, "favorite": ["Apple", "Orange"]}
  dumpedJson = json.dumps(src)
  assert "{\"name\": \"Mike\", \"age\": 25, \"favorite\": [\"Apple\", \"Orange\"]}" == dumpedJson
  dist = json.loads(dumpedJson)
  assert dist == src

def test_generator():
  def counter(max):
    i = 0
    while i < max:
      p = (yield i)  
      if p is None:
        i += 1
      else:
        i = p #sendで受け取った値で更新する。
  
  ctr = counter(10)
  next(ctr)
  next(ctr)
  a = next(ctr)
  assert a == 2
  ctr.send(7)
  b = next(ctr)
  assert b == 8

def test_itemgetter():
  sample = [('Foo', 23), ('Bar', 19), ('Baz', 32)]
  result = sorted(sample, key=itemgetter(1))
  assert result == [('Bar', 19), ('Foo', 23), ('Baz', 32)]

def test_attrgetter():
  class Student():
    def __init__(self, name, score) -> None:
      self.name = name
      self.score = score

    def __eq__(self, __value: object) -> bool:
      if not isinstance(__value, Student):
        return False
      return self.name == __value.name and self.score == __value.score

  students = [Student('Joe', 78), Student('Taro', 56), Student('Mike', 56)]
  # attrgetterの第2引数でも降順でソートしたい場合は別にもう一回sortedを呼び出すしかないのか？
  result = sorted(students, key=attrgetter('score', 'name'), reverse=True)
  assert result == [Student('Joe', 78), Student('Taro', 56), Student('Mike', 56)]

def test_key_function():
  fruits = ['banana', 'apple', 'orange']
  items = {'apple': 220, 'orange': 200, 'banana': 180}
  #itemsの値に従ってソートしてくれる。
  result = sorted(fruits, key=items.__getitem__)
  assert result == ['banana', 'orange', 'apple']

def test_unicode_category():
  sample = '𩸽を𠮟る𠮷野家と髙﨑〜'
  print()
  for i, c in enumerate(sample):
    print(i, c, unicodedata.category(c), end=' ')
    print(unicodedata.name(c))  

MySampleCode: TypeAlias = list[int]

def test_type_alias():
  #type文は3.12から使用可能。
  #type Id = [[str], [int], [str]]

  def create_sample_code(values: [int]) -> MySampleCode:
    return MySampleCode(values)

  result = create_sample_code([1, 2, 3])
  #あくまでも別名であって元の型のオブジェクトとの比較結果が変わるわけではない。
  assert result == [1, 2, 3] 
