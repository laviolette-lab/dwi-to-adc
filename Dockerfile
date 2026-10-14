FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --locked --release --bin dwi-to-adc

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 app
COPY --from=build /src/target/release/dwi-to-adc /usr/local/bin/dwi-to-adc
USER 10001:10001
ENTRYPOINT ["dwi-to-adc"]
