#!/usr/bin/python3
# -*- coding: utf-8 -*-

import mysql.connector

def get_sample_students() -> list[any]:
  db = mysql.connector.connect(
      host="localhost",
      user="sampleuser",
      password="samplepass",
      database="test"
  )
  cursor = db.cursor()
  cursor.execute("SELECT * FROM students")
  results = cursor.fetchall()
  return results

if __name__ == '__main__':
    print("database module load")
