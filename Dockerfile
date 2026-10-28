FROM registry.access.redhat.com/ubi10/ubi:latest AS builder

RUN dnf install -y gcc gcc-c++ make rust cargo glibc-static libstdc++-static && \
    dnf clean all

WORKDIR /build
COPY . .

ENV RUSTFLAGS="-C target-feature=+crt-static -C strip=symbols"
RUN cargo build \
    --release \
    --package tempura \
    --target x86_64-unknown-linux-gnu

FROM scratch

COPY --from=builder /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /build/target/x86_64-unknown-linux-gnu/release/tempura /usr/local/bin/tempura

EXPOSE 8080
USER 65534

ENTRYPOINT ["/usr/local/bin/tempura"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
