import argparse
import json
import os
import ssl
from .owner import Owner
from .server import make_server
from .. import __version__


def main():
    parser = argparse.ArgumentParser(description="MuNet durable training swarm owner")
    parser.add_argument("--version", action="version", version="MuNet server " + __version__)
    parser.add_argument("directory", help="job directory produced by create_job")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lease-seconds", type=float, default=120)
    parser.add_argument("--target-seconds", type=float, default=30)
    parser.add_argument("--max-bundle", type=int, default=4)
    parser.add_argument("--cert", help="PEM certificate for direct HTTPS")
    parser.add_argument("--key", help="PEM private key for direct HTTPS")
    parser.add_argument("--allow-http", action="store_true", help="allow plaintext on a trusted private network")
    parser.add_argument("--status", action="store_true", help="inspect a stopped owner's journal")
    args = parser.parse_args()
    if bool(args.cert) != bool(args.key):
        parser.error("--cert and --key must be supplied together")
    if not args.status and args.bind not in ("127.0.0.1", "localhost", "::1") and not args.cert and not args.allow_http:
        parser.error("non-loopback listeners require TLS or explicit --allow-http")
    owner = Owner(args.directory, lease_seconds=args.lease_seconds,
                  target_seconds=args.target_seconds, max_bundle=args.max_bundle)
    server = None
    try:
        if args.status:
            print(json.dumps(owner.status(), indent=2))
            return
        server = make_server(owner, (args.bind, args.port), os.environ.get("MUNET_SWARM_TOKEN", ""))
        if args.cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(args.cert, args.key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f"MuNet owner ready at {'https' if args.cert else 'http'}://{args.bind}:{server.server_port}", flush=True)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        owner.close()


if __name__ == "__main__":
    main()
