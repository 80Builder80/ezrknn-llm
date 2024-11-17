FROM ubuntu:focal

# Copy setup scripts
COPY setup.sh /setup.sh
COPY entrypoint.sh /entrypoint.sh

# Run setup
RUN chmod +x /setup.sh /entrypoint.sh && ./setup.sh

# Define a volume for models
VOLUME /models

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
