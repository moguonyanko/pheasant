import pheasant.database as db

def test_get_sample_students():
  results = db.get_sample_students()
  print(results)
  assert len(results) > 0
 
def test_select_now():
  current_time = db.select_now()
  print(current_time)
  assert current_time is not None
