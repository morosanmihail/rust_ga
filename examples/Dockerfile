FROM rust:1-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin maze_server

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/maze_server /usr/local/bin/maze_server
EXPOSE 3000
CMD ["maze_server"]
