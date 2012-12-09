#!/usr/bin/python3
# -*- coding: utf-8 -*

def application(environ, start_response):
	start_response('200 OK', [('Content-type', 'text/plain')])
	return 'WelCome, Pheasant Public Searvice!'

from wsgiref import simple_server

if __name__ == '__main__':
	server = simple_server.make_server('', 8080, application)
	server.serve_forever()

