import pheasant.database as db

def test_get_sample_students():
  results = db.get_sample_students()
  print(results)
  assert len(results) > 0
 