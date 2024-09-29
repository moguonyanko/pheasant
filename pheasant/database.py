#!/usr/bin/python3
# -*- coding: utf-8 -*-

import mysql.connector
import psycopg2

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

def select_now() -> str:
  conn = psycopg2.connect(
      database="postgres",
      user="myuser",
      password="mypass",
      host="localhost",
      port="5432"
  )   
  with conn.cursor() as cursor:
      cursor.execute("SELECT NOW()")
      row = cursor.fetchone()
      current_time = row[0]

  conn.close()  
  return current_time

if __name__ == '__main__':
    print(select_now())
    print("database module load")
