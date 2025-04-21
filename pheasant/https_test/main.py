import ssl
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

"""
## 参考

Udemy：【ハンズオン】ネットワークセキュリティ入門講座
https://tokyo-gas-dx.udemy.com/course/nsbjbysuzuki/learn/lecture/39787790#overview
"""

# generate certs first:
# openssl req -new -x509 -keyout cert.pem -out cert.pem -days 365 -nodes
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

cert_path = f"{Path.home()}/etc/cert/cert.pem"
key_path = f"{Path.home()}/etc/cert/key.pem"

context.load_cert_chain(cert_path, key_path)
with HTTPServer(("0.0.0.0", 8443), SimpleHTTPRequestHandler) as myhttpd: 
	myhttpd.socket = context.wrap_socket(myhttpd.socket, server_side=True) 
	myhttpd.serve_forever()
