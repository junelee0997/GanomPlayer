import socket
import activate
import json
# socket details
IP = '127.0.0.1'
PORT = 25566
SIZE = 4096

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((IP, PORT))
server_socket.listen()
print(f"Server start in {IP}")

try:
    while True:
                ch = 1
                print("Ready to accept")
                client_socket, client_addr = server_socket.accept()
                client_socket.send("start\n".encode("utf-8"))
                while True:
                    msg = client_socket.recv(SIZE)
                    if not msg:
                        print("Ended the connection")
                        ch = 0
                        break
                    if msg == b'-1':
                        client_socket.send("Train pended\n".encode())
                        print("Client pended the connection")
                        break
                    msg = eval(msg.decode("utf-8"))
                    print(msg)
                    recv = json.dumps(activate.run(msg)) + '\n'
                    client_socket.send(recv.encode("utf-8"))
                if not ch:
                    break

except Exception as e:
        raise e

finally:
        print("done")
        server_socket.close()
