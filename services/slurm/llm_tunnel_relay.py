#!/usr/bin/env python3
"""TCP relay: docker bridge gateway -> host loopback SSH tunnel.

Bridges the docker network to an SSH RemoteForward that binds only to the
host loopback (the default when sshd `GatewayPorts` cannot be changed).

    container --(GATEWAY_HOST:PORT via extra_hosts host-gateway)-->
    host BRIDGE_IP:PORT (this relay) --> 127.0.0.1:PORT (ssh -R tunnel) -->
    laptop --> GATEWAY_HOST:443

Usage:
    python3 services/slurm/llm_tunnel_relay.py            # defaults below
    python3 services/slurm/llm_tunnel_relay.py --bind 172.17.0.1 --port 9443

Find BRIDGE_IP with `ip addr show docker0` (default bridge is 172.17.0.1).
"""

from __future__ import annotations

import argparse
import socket
import threading


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle(client: socket.socket, target: tuple[str, int]) -> None:
    try:
        upstream = socket.create_connection(target)
    except OSError as exc:
        client.close()
        print(f"  upstream connect failed {target}: {exc}", flush=True)
        return
    threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="172.17.0.1", help="docker bridge gateway IP to listen on")
    parser.add_argument("--port", type=int, default=9443, help="port (same on both ends)")
    parser.add_argument("--target-host", default="127.0.0.1", help="ssh tunnel host")
    parser.add_argument("--target-port", type=int, default=None, help="ssh tunnel port (default: --port)")
    args = parser.parse_args()

    target = (args.target_host, args.target_port or args.port)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(128)
    print(f"relay listening on {args.bind}:{args.port} -> {target[0]}:{target[1]}", flush=True)

    try:
        while True:
            client, _ = server.accept()
            _handle(client, target)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
