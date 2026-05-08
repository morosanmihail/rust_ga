# Examples

## Docker

Build from repo root (workspace context required):

```bash
docker build -f examples/Dockerfile -t rust-ga-server .
```

Run:

```bash
docker run -p 3000:3000 rust-ga-server
```

Server listens on port 3000.
