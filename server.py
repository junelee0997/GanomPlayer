import socket
import json
#import time
# socket details
IP = '127.0.0.1'
PORT = 25566
SIZE = 4096


def loop(activate, activate2):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP, PORT))
    server_socket.listen()
    print(f"Server start in {IP}")

    while True:
        ch = 1
        #print("Ready to accept")
        client_socket, client_addr = server_socket.accept()
        # client_socket.send("start\n".encode("utf-8"))
        while True:
            #s = time.time()
            try:
                msg = client_socket.recv(SIZE)
            except ConnectionAbortedError:
                print("Connection Aborted")
                break
            print("msg:", msg)
            #s2 = time.time()
            if not msg:
                continue
            if msg == b'-1':
                client_socket.send("Train pended\n".encode())
                print("Client pended the connection")
                break
            try:
                msg = json.loads(msg.decode("utf-8"))
                recv = json.dumps(activate(msg)) + '\n'
                #s3 = time.time()
                client_socket.send(recv.encode("utf-8"))
                #s4 = time.time()
            except json.decoder.JSONDecodeError:
                print("parsing failed")
            #t1 = s2 - s
            #t2 = time.time() - s2
            #print(f'Time elapsed: {t1:.3f} {t2:.3f} {s4 - s3:.3f} {t1 + t2:.3f}', )
            # print("check")
        activate2()
        if not ch:
            break

    print("done")
    server_socket.close()
